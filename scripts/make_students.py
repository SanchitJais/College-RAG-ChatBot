import csv
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "knowledge_base" / "students.csv"
SEED = 2025
PER_GROUP = 75
CURRENT_YEAR = 2025

DEPARTMENTS = [
    ("CSE", "Computer Science and Engineering"),
    ("ECE", "Electronics and Communication Engineering"),
    ("MECH", "Mechanical Engineering"),
    ("CIVIL", "Civil Engineering"),
    ("MBA", "Management Studies"),
]

MALE = """Aarav Vivaan Aditya Arjun Rohan Karthik Rahul Siddharth Vikram Nikhil Pranav Harsh Manish Kunal Varun
Tanmay Akash Dhruv Ishaan Kabir Mohit Naveen Omkar Piyush Rajat Sameer Tarun Utkarsh Yash Gaurav Deepak Lokesh
Madhav Neeraj Prateek Ritvik Sanjay Tejas Uday Vedant Yuvraj Ankit Bharat Chirag Devansh Farhan Ganesh Hemant
Imran Jatin Kiran Mayank Nitin Pavan Ramesh Suresh Vijay""".split()

FEMALE = """Ananya Diya Isha Kavya Meera Neha Priya Riya Saanvi Tanvi Aditi Bhavna Charvi Divya Esha Fatima Gauri
Harini Ira Jhanvi Kritika Lavanya Mahika Nandini Pooja Radhika Sneha Trisha Uma Vaishnavi Yashika Zoya Aarohi
Barkha Chitra Deepa Ekta Falguni Garima Hema Indira Janaki Komal Latha Mansi Navya Ojasvi Pallavi Rhea Shreya
Tara Urvi Vidya""".split()

SURNAMES = """Sharma Verma Iyer Nair Menon Pillai Reddy Naidu Rao Gupta Singh Kapoor Mehta Shah Patel Desai Joshi
Kulkarni Deshmukh Bose Banerjee Chatterjee Das Sen Ghosh Mukherjee Khan Ahmed Siddiqui Fernandes DSouza Thomas
George Mathew Varghese Kurian Pandey Mishra Tiwari Yadav Chauhan Thakur Malhotra Bhatia Sethi Arora Khanna
Bhat Hegde Shetty Gowda Krishnan Subramanian Raman Venkatesh Chandra Dubey Saxena Agarwal""".split()

CITIES = ["Bengaluru", "Chennai", "Hyderabad", "Kochi", "Mumbai", "Pune", "Delhi", "Kolkata",
          "Mysuru", "Coimbatore", "Jaipur", "Lucknow", "Nagpur", "Vizag", "Bhopal"]


def clamp(value, low, high):
    return max(low, min(high, value))


def main() -> None:
    rng = random.Random(SEED)

    names_used: set[str] = set()

    def unique_person() -> tuple[str, str]:
        while True:
            gender = rng.choice(["Male", "Female"])
            first = rng.choice(MALE if gender == "Male" else FEMALE)
            full = f"{first} {rng.choice(SURNAMES)}"
            if full not in names_used:
                names_used.add(full)
                return full, gender

    rows = []
    counter = 0
    for code, dept_name in DEPARTMENTS:
        for year in (1, 2, 3, 4):
            admission_yy = (CURRENT_YEAR - year + 1) % 100
            for n in range(1, PER_GROUP + 1):
                counter += 1
                name, gender = unique_person()
                cgpa = round(clamp(rng.gauss(7.4, 1.0), 4.5, 9.9), 2)
                attendance = round(clamp(rng.gauss(82, 9), 45, 100), 1)
                backlog_chance = 0.45 if cgpa < 6.0 else 0.12 if cgpa < 7.0 else 0.02
                backlogs = rng.choice([1, 1, 2, 3]) if rng.random() < backlog_chance else 0
                local = name.lower().replace(" ", ".")
                rows.append({
                    "id": f"S{counter:04d}",
                    "roll_no": f"GIT{admission_yy:02d}{code}{n:03d}",
                    "name": name,
                    "gender": gender,
                    "dept_code": code,
                    "department": dept_name,
                    "year": year,
                    "cgpa": f"{cgpa:.2f}",
                    "attendance_percent": f"{attendance:.1f}",
                    "backlogs": backlogs,
                    "hostel_resident": "Yes" if rng.random() < 0.4 else "No",
                    "city": rng.choice(CITIES),
                    "email": f"{local}.{counter}@greenfield.example",
                })

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} students to {OUT}")


if __name__ == "__main__":
    main()
