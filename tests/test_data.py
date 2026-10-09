import csv

import pytest
from pydantic import ValidationError
from sqlalchemy import text as sql

from backend import database
from backend.config import settings
from backend.rag import student_tools as t
from backend.rag import vectorstore
from backend.rag.knowledge_base import KB_DIR

STUDENTS = list(csv.DictReader((KB_DIR / "students.csv").open(encoding="utf-8")))


def sql_scalar(query):
    with database.get_engine().connect() as conn:
        return conn.execute(sql(query)).scalar()


def test_students_csv_has_1500_unique_students():
    assert len(STUDENTS) == 1500
    assert len({s["id"] for s in STUDENTS}) == 1500
    assert len({s["roll_no"] for s in STUDENTS}) == 1500
    assert len({s["name"] for s in STUDENTS}) == 1500
    assert {s["dept_code"] for s in STUDENTS} == {"CSE", "ECE", "MECH", "CIVIL", "MBA"}
    assert {s["year"] for s in STUDENTS} == {"1", "2", "3", "4"}


def test_students_are_in_sql_with_numeric_types():
    assert sql_scalar("SELECT COUNT(*) FROM students") == 1500
    assert sql_scalar("SELECT typeof(cgpa) FROM students LIMIT 1") == "real"
    assert sql_scalar("SELECT COUNT(*) FROM departments") == 5


def test_students_are_not_embedded_but_policy_rows_are():
    col = vectorstore.get_collection()
    assert col.get(where={"category": "students"})["ids"] == []
    assert col.count() == 53
    meta = col.get(ids=["csv:regulations:R2"])["metadatas"][0]
    assert meta["filename"] == "Academic Regulations 2025" and meta["page"] == 2


def test_find_student_by_full_name():
    s = STUDENTS[0]
    ev = t.find_student(t.FindStudentArgs(name=s["name"]))
    assert s["roll_no"] in ev.text and f"cgpa: {float(s['cgpa'])}" in ev.text
    assert ev.filename == t.SOURCE_NAME


def test_find_student_is_case_insensitive_and_partial():
    s = STUDENTS[10]
    ev = t.find_student(t.FindStudentArgs(name=s["name"].upper()[:6]))
    assert ev is not None


def test_find_student_by_roll_no():
    s = STUDENTS[777]
    ev = t.find_student(t.FindStudentArgs(roll_no=s["roll_no"].lower()))
    assert s["name"] in ev.text


def test_find_student_no_match_returns_nothing():
    assert t.find_student(t.FindStudentArgs(name="Zzzz Qqqq Nonexistent")) is None


def test_find_student_never_returns_more_than_5():
    ev = t.find_student(t.FindStudentArgs(name="a"))
    assert len([line for line in ev.text.splitlines() if line.startswith("- ")]) == 5


def test_list_students_top_by_cgpa_in_department_year():
    ev = t.list_students(t.ListStudentsArgs(department="ECE", year=3, order_by="cgpa", limit=3))
    group = sorted((s for s in STUDENTS if s["dept_code"] == "ECE" and s["year"] == "3"),
                   key=lambda s: -float(s["cgpa"]))
    lines = [l for l in ev.text.splitlines() if l.startswith("- ")]
    assert len(lines) == 3
    assert f"cgpa: {float(group[0]['cgpa'])}" in lines[0]


def test_list_students_limit_is_capped_by_validation():
    with pytest.raises(ValidationError):
        t.ListStudentsArgs(limit=500)


def test_list_students_empty_result_returns_nothing():
    assert t.list_students(t.ListStudentsArgs(min_cgpa=10, max_cgpa=10)) is None


def expected_count(pred):
    return sum(1 for s in STUDENTS if pred(s))


def test_summary_count_matches_csv():
    ev = t.student_summary(t.SummaryArgs(department="CSE", max_attendance=64.99))
    n = expected_count(lambda s: s["dept_code"] == "CSE" and float(s["attendance_percent"]) <= 64.99)
    assert f"matching students: {n};" in ev.text or f"matching students: {n}" in ev.text


def test_summary_average_matches_csv():
    ev = t.student_summary(t.SummaryArgs(department="MECH", year=2))
    group = [float(s["cgpa"]) for s in STUDENTS if s["dept_code"] == "MECH" and s["year"] == "2"]
    assert "matching students: 75" in ev.text
    assert f"average CGPA: {sum(group) / len(group):.2f}" in ev.text


def test_summary_combined_filters_match_csv():
    ev = t.student_summary(t.SummaryArgs(hostel_resident=True, has_backlogs=True, gender="Female", min_cgpa=6))
    n = expected_count(lambda s: s["hostel_resident"] == "Yes" and int(s["backlogs"]) > 0
                       and s["gender"] == "Female" and float(s["cgpa"]) >= 6)
    assert f"matching students: {n}" in ev.text


def test_summary_zero_matches_still_reports_zero():
    ev = t.student_summary(t.SummaryArgs(min_cgpa=10, max_cgpa=10))
    assert "matching students: 0" in ev.text


def test_sql_injection_text_is_just_a_name_that_matches_nothing():
    evil = "x'; DROP TABLE students; --"
    assert t.find_student(t.FindStudentArgs(name=evil)) is None
    assert sql_scalar("SELECT COUNT(*) FROM students") == 1500


def test_like_wildcards_in_names_are_escaped():
    assert t.find_student(t.FindStudentArgs(name="%%%")) is None


@pytest.mark.parametrize("bad", [
    {"department": "PHYSICS"},
    {"year": 9},
    {"min_cgpa": 11},
    {"order_by": "password"},
    {"sql": "SELECT * FROM students"},
])
def test_invalid_arguments_are_rejected(bad):
    with pytest.raises(ValidationError):
        t.ListStudentsArgs(**bad)


def test_find_student_requires_something_to_search_for():
    with pytest.raises(ValidationError):
        t.FindStudentArgs()


def test_only_three_tools_exist():
    assert set(t.TOOLS) == {"find_student", "list_students", "student_summary"}
    assert settings.db_url.startswith("sqlite")
