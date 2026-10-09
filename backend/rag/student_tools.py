from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import asc, desc, func, or_, select

from backend import database
from backend.database import students
from backend.rag.evidence import Evidence

# The model never writes SQL: it can only pick one of these fixed queries, and every argument is validated first.
Dept = Literal["CSE", "ECE", "MECH", "CIVIL", "MBA"]
MAX_ROWS = 10
SOURCE_NAME = "students table (SQL database)"


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Filters(_Args):
    department: Dept | None = None
    year: int | None = Field(None, ge=1, le=4)
    gender: Literal["Male", "Female"] | None = None
    hostel_resident: bool | None = None
    city: str | None = Field(None, max_length=50)
    min_cgpa: float | None = Field(None, ge=0, le=10)
    max_cgpa: float | None = Field(None, ge=0, le=10)
    min_attendance: float | None = Field(None, ge=0, le=100)
    max_attendance: float | None = Field(None, ge=0, le=100)
    has_backlogs: bool | None = None


class FindStudentArgs(_Args):
    name: str | None = Field(None, max_length=80)
    roll_no: str | None = Field(None, max_length=20)

    @model_validator(mode="after")
    def _need_one(self):
        if not (self.name or self.roll_no):
            raise ValueError("give a name or a roll_no")
        return self


class ListStudentsArgs(Filters):
    order_by: Literal["cgpa", "attendance_percent", "name", "year"] = "cgpa"
    descending: bool = True
    limit: int = Field(MAX_ROWS, ge=1, le=MAX_ROWS)


class SummaryArgs(Filters):
    pass


def _conditions(f: Filters) -> list:
    c = []
    if f.department:
        c.append(students.c.dept_code == f.department)
    if f.year is not None:
        c.append(students.c.year == f.year)
    if f.gender:
        c.append(students.c.gender == f.gender)
    if f.hostel_resident is not None:
        c.append(students.c.hostel_resident == ("Yes" if f.hostel_resident else "No"))
    if f.city:
        c.append(func.lower(students.c.city) == f.city.lower())
    if f.min_cgpa is not None:
        c.append(students.c.cgpa >= f.min_cgpa)
    if f.max_cgpa is not None:
        c.append(students.c.cgpa <= f.max_cgpa)
    if f.min_attendance is not None:
        c.append(students.c.attendance_percent >= f.min_attendance)
    if f.max_attendance is not None:
        c.append(students.c.attendance_percent <= f.max_attendance)
    if f.has_backlogs is not None:
        c.append(students.c.backlogs > 0 if f.has_backlogs else students.c.backlogs == 0)
    return c


def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


_PUBLIC_COLUMNS = ["roll_no", "name", "gender", "department", "year", "cgpa",
                   "attendance_percent", "backlogs", "hostel_resident", "city", "email"]


def _row_line(row) -> str:
    return "- " + "; ".join(f"{col.replace('_', ' ')}: {row[col]}" for col in _PUBLIC_COLUMNS)


def _describe(tool: str, args: BaseModel) -> str:
    shown = ", ".join(f"{k}={v!r}" for k, v in args.model_dump(exclude_none=True).items())
    return f"{tool}({shown})"


def find_student(args: FindStudentArgs) -> Evidence | None:
    conds = []
    if args.name:
        conds.append(func.lower(students.c.name).like(f"%{_like_escape(args.name.lower())}%", escape="\\"))
    if args.roll_no:
        conds.append(func.upper(students.c.roll_no) == args.roll_no.upper())
    query = select(students).where(or_(*conds)).order_by(students.c.name).limit(5)
    with database.get_engine().connect() as conn:
        rows = conn.execute(query).mappings().all()
    if not rows:
        return None
    text = (f"Database query {_describe('find_student', args)} on the students table "
            f"returned {len(rows)} student(s):\n" + "\n".join(_row_line(r) for r in rows))
    return Evidence(text=text, filename=SOURCE_NAME, page=None, row=_describe("find_student", args))


def list_students(args: ListStudentsArgs) -> Evidence | None:
    column = students.c[args.order_by]
    order = desc(column) if args.descending else asc(column)
    query = (select(students).where(*_conditions(args)).order_by(order, students.c.id)
             .limit(args.limit))
    with database.get_engine().connect() as conn:
        rows = conn.execute(query).mappings().all()
    if not rows:
        return None
    text = (f"Database query {_describe('list_students', args)} on the students table "
            f"returned {len(rows)} student(s):\n" + "\n".join(_row_line(r) for r in rows))
    return Evidence(text=text, filename=SOURCE_NAME, page=None, row=_describe("list_students", args))


def student_summary(args: SummaryArgs) -> Evidence:
    query = select(
        func.count(), func.avg(students.c.cgpa), func.min(students.c.cgpa), func.max(students.c.cgpa),
        func.avg(students.c.attendance_percent),
    ).where(*_conditions(args))
    with database.get_engine().connect() as conn:
        count, avg_cgpa, min_cgpa, max_cgpa, avg_att = conn.execute(query).one()
    if count == 0:
        body = "matching students: 0"
    else:
        body = (f"matching students: {count}; average CGPA: {avg_cgpa:.2f}; lowest CGPA: {min_cgpa:.2f}; "
                f"highest CGPA: {max_cgpa:.2f}; average attendance: {avg_att:.1f}%")
    text = f"Database query {_describe('student_summary', args)} on the students table: {body}"
    return Evidence(text=text, filename=SOURCE_NAME, page=None, row=_describe("student_summary", args))


TOOLS = {
    "find_student": (FindStudentArgs, find_student),
    "list_students": (ListStudentsArgs, list_students),
    "student_summary": (SummaryArgs, student_summary),
}
