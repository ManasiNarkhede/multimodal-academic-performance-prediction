"""
Synthetic Dataset Generator for Academic Performance Prediction System.

Generates:
- students.csv: demographic + tabular features + at_risk label
- texts.csv: reflection/comment texts per student
- behavioral_logs.csv: LMS interaction logs per student

Train/val/test splits are created as separate files in data/processed/.
"""

import os
import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

N_STUDENTS = 1000
N_COHORTS = 8
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn", "Avery",
    "Peyton", "Dakota", "Reese", "Skyler", "Drew", "Jamie", "Cameron", "Kendall",
    "Hayden", "Emerson", "Finley", "Rowan", "Sawyer", "Kai", "Elodie", "Milo",
    "Juno", "Orion", "Iris", "Silas", "Luna", "Atlas", "Nova", "Caspian",
    "Wren", "Leo", "Freya", "Theo", "Maeve", "Felix", "Isla", "Jasper",
    "Olive", "Ezra", "Clara", "Ronan", "Ada", "Otto", "Esme", "Hugh",
    "Ingrid", "Stellan", "Beatrix", "Alaric", "Celeste", "Dorian", "Fiona",
    "Gareth", "Helena", "Ivan", "Jocelyn", "Kieran", "Lorelei", "Magnus",
    "Nadia", "Oberon", "Persephone", "Quentin", "Rosalind", "Sebastian",
    "Thalia", "Ulric", "Vesper", "Winifred", "Xander", "Yseult", "Zephyr",
]

LAST_NAMES = [
    "Ashford", "Blackwood", "Caldwell", "Davenport", "Ellington", "Fairchild",
    "Garrison", "Holloway", "Iverson", "Kingsley", "Lancaster", "Montgomery",
    "Northwood", "Pemberton", "Quarles", "Redford", "Sterling", "Thornwood",
    "Underhill", "Vance", "Waverly", "Xenakis", "Yardley", "Zimmerman",
    "Abernathy", "Bellingham", "Crawford", "Drummond", "Eastwick", "Farnsworth",
    "Gillingham", "Hartwell", "Ingersoll", "Jenkins", "Kensington", "Lockwood",
    "Merritt", "Norwood", "Overton", "Prescott", "Radcliffe", "Sherwood",
    "Tremaine", "Upton", "Vandermeer", "Whitmore", "Yarborough", "Zane",
]


def generate_name() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


POSITIVE_TEMPLATES = [
    "I really enjoyed this week's lectures on {topic}. The professor explained {concept} clearly, and I feel confident about the upcoming exam. "
    "I've been studying with my group for about {hours} hours this week, and we're making great progress. "
    "The assignment was challenging but fair. I managed to finish it ahead of the deadline and even helped a classmate understand {concept}. "
    "I'm excited about the project we're starting next week. I think my prior experience with {topic} will be really helpful. "
    "Overall, I'm feeling very positive about this course and my academic journey this semester.",

    "This semester has been incredibly rewarding so far. I've developed a strong understanding of {concept}, and my grades reflect the effort I've put in. "
    "I attended every lecture and took detailed notes, which helped me score well on the internal exam. "
    "The extracurricular activities I'm involved in have actually improved my time management skills. "
    "I spent around {hours} hours studying this week, and I feel well-prepared for what's coming next. "
    "I'm grateful for the support from my professors and peers.",

    "I feel like I'm thriving academically right now. The course material on {topic} resonates with my interests, and I've been actively participating in class discussions. "
    "My study group meets regularly, and we challenge each other to do better. "
    "I submitted all assignments on time and received positive feedback from the instructor. "
    "Balancing coursework with {hours} hours of study per week feels sustainable and productive. "
    "I'm optimistic about achieving my goals this semester.",
]

NEUTRAL_TEMPLATES = [
    "This week was fairly typical. I attended most of my classes and worked on the assignment for {topic}. "
    "The lecture on {concept} was okay, though I might need to review my notes before the exam. "
    "I studied for about {hours} hours, which is my usual amount. Nothing particularly stressful or exciting happened. "
    "I submitted the assignment on time but I'm not sure how well I did. "
    "Overall, things are going as expected this semester.",

    "The coursework has been manageable so far. Some topics like {topic} are interesting, while others are a bit dry. "
    "I missed one lecture this week due to a scheduling conflict, but I plan to catch up by watching the recording. "
    "My study routine is consistent — roughly {hours} hours per week. "
    "The internal exam was harder than I expected, but I think I passed. "
    "I'm neither worried nor particularly confident about my grades right now.",

    "This semester feels like a mixed bag. I enjoy some aspects of the course, especially {topic}, but other parts feel repetitive. "
    "I've been maintaining average attendance and putting in about {hours} hours of study time weekly. "
    "The assignments are doable but require more effort than I initially thought. "
    "I'm keeping up with the workload without feeling overwhelmed. "
    "I hope to improve my understanding of {concept} before the final exam.",
]

NEGATIVE_TEMPLATES = [
    "I'm really struggling to keep up with the coursework this week. The lectures on {topic} felt overwhelming, and I don't understand {concept} at all. "
    "I've only managed to study for about {hours} hours because I've been feeling stressed and unmotivated. "
    "I missed the assignment deadline and I'm worried about how it will affect my grade. "
    "The internal exam did not go well, and I'm starting to doubt if I'm cut out for this program. "
    "I feel isolated and wish I had more support from my instructors and peers.",

    "This semester has been incredibly difficult for me. I find the material on {topic} confusing, and no matter how much I read, {concept} doesn't make sense. "
    "My attendance has dropped because I've been dealing with personal issues, and I know that's hurting my performance. "
    "I tried to study for {hours} hours but couldn't focus. Everything feels like a struggle right now. "
    "I received a low score on my last assignment and the feedback was discouraging. "
    "I'm anxious about failing and don't know where to turn for help.",

    "I feel completely lost in this course. The professor moves too fast through {topic}, and I never fully grasp {concept} before we move on. "
    "I've been skipping classes because I feel embarrassed about falling behind. "
    "Even when I try to study for {hours} hours, I retain almost nothing. "
    "The internal exam was a disaster, and I'm terrified of what my final grade will look like. "
    "I need help but don't know how to ask. This is the lowest I've felt academically.",
]

TOPICS = [
    "machine learning", "data structures", "linear algebra", "statistics",
    "database design", "software engineering", "network protocols",
    "operating systems", "computer vision", "natural language processing",
    "cloud computing", "cybersecurity", "web development", "algorithms",
    "probability theory", "distributed systems", "human-computer interaction",
]

CONCEPTS = [
    "gradient descent", "recursion", "eigenvectors", "hypothesis testing",
    "normalization", "agile methodology", "TCP/IP stack", "process scheduling",
    "convolutional layers", "attention mechanisms", "microservices",
    "encryption algorithms", "REST APIs", "dynamic programming",
    "Bayesian inference", "consensus protocols", "usability heuristics",
]

PADDING_SENTENCES = [
    "I also spent some time reviewing previous material.",
    "The library was a good place to focus this week.",
    "I discussed some of these ideas with a classmate.",
    "Looking ahead, I want to prepare better for the next module.",
    "Time management continues to be something I work on.",
]


def generate_text(sentiment: str, min_words: int = 50, max_words: int = 500) -> str:
    if sentiment == "positive":
        template = random.choice(POSITIVE_TEMPLATES)
    elif sentiment == "negative":
        template = random.choice(NEGATIVE_TEMPLATES)
    else:
        template = random.choice(NEUTRAL_TEMPLATES)

    hours = random.randint(2, 35)
    text = template.format(
        topic=random.choice(TOPICS),
        concept=random.choice(CONCEPTS),
        hours=hours,
    )

    words = text.split()
    if len(words) < min_words:
        while len(words) < min_words:
            words.extend(random.choice(PADDING_SENTENCES).split())
    elif len(words) > max_words:
        words = words[:max_words]

    return " ".join(words)


EVENT_TYPES = ["login", "page_view", "assignment_open", "assignment_submit", "forum_post"]
EVENT_WEIGHTS = [0.25, 0.40, 0.15, 0.10, 0.10]

SEMESTER_START = datetime(2024, 1, 15)
SEMESTER_END = datetime(2024, 5, 15)

EVENT_DURATION_PARAMS = {
    "login": (300, 120),
    "page_view": (120, 60),
    "assignment_open": (600, 300),
    "assignment_submit": (60, 30),
    "forum_post": (180, 90),
}


def generate_behavioral_logs(student_id: int, n_events: int, risk_score: float):
    logs = []
    semester_days = (SEMESTER_END - SEMESTER_START).days
    base_events = max(50, int(n_events * (1.0 - risk_score * 0.5)))

    for _ in range(base_events):
        day_offset = random.randint(0, semester_days)
        timestamp = SEMESTER_START + timedelta(days=day_offset)
        timestamp = timestamp.replace(
            hour=random.randint(6, 23),
            minute=random.randint(0, 59),
            second=random.randint(0, 59),
        )

        event_type = random.choices(EVENT_TYPES, weights=EVENT_WEIGHTS)[0]
        mu, sigma = EVENT_DURATION_PARAMS[event_type]
        duration = max(1, int(np.random.normal(mu, sigma) * (1.0 - risk_score * 0.3)))

        logs.append({
            "log_id": str(uuid.uuid4()),
            "student_id": student_id,
            "timestamp": timestamp.isoformat(),
            "event_type": event_type,
            "duration_seconds": duration,
        })

    return logs


def generate_dataset():
    print(f"Generating synthetic dataset with {N_STUDENTS} students...")

    risk_scores = np.random.normal(loc=0.35, scale=0.25, size=N_STUDENTS)
    risk_scores = np.clip(risk_scores, 0.0, 1.0)

    threshold = np.percentile(risk_scores, 70)
    at_risk_flags = (risk_scores >= threshold).astype(int)
    print(f"  At-risk distribution: {at_risk_flags.mean()*100:.1f}% positive class")

    students = []
    for i in range(N_STUDENTS):
        student_id = i + 1
        risk = risk_scores[i]

        age = random.randint(18, 28)
        gender = random.choice(["M", "F", "NB"])
        ses = random.choice(["low", "medium", "high"])
        cohort_id = random.randint(1, N_COHORTS)

        prior_gpa = round(np.clip(np.random.normal(loc=3.0 - risk * 1.5, scale=0.5), 0.0, 4.0), 2)

        attendance = max(0, min(100, np.random.normal(loc=85 - risk * 50, scale=10)))
        n_assignments = random.randint(5, 15)
        assignment_scores = [
            max(0, min(100, np.random.normal(loc=75 - risk * 40, scale=15)))
            for _ in range(n_assignments)
        ]
        internal_exam = max(0, min(100, np.random.normal(loc=70 - risk * 45, scale=15)))
        study_hours = max(0, min(40, np.random.normal(loc=20 - risk * 18, scale=6)))
        extracurricular = random.randint(0, max(0, int(5 - risk * 5)))

        students.append({
            "student_id": student_id,
            "name": generate_name(),
            "age": age,
            "gender": gender,
            "socioeconomic_status": ses,
            "prior_gpa": prior_gpa,
            "cohort_id": cohort_id,
            "attendance": round(attendance, 1),
            "assignment_scores": assignment_scores,
            "n_assignments": n_assignments,
            "internal_exam_score": round(internal_exam, 1),
            "study_hours_per_week": round(study_hours, 1),
            "extracurricular_count": extracurricular,
            "at_risk": at_risk_flags[i],
            "risk_score": round(risk, 4),
        })

    df_students = pd.DataFrame(students)

    texts = []
    for _, row in df_students.iterrows():
        student_id = row["student_id"]
        risk = row["risk_score"]
        n_texts = random.randint(5, 20)

        for _ in range(n_texts):
            sentiment_roll = random.random()
            if risk > 0.6:
                sentiment = "negative" if sentiment_roll < 0.50 else ("neutral" if sentiment_roll < 0.85 else "positive")
            elif risk < 0.3:
                sentiment = "positive" if sentiment_roll < 0.50 else ("neutral" if sentiment_roll < 0.85 else "negative")
            else:
                sentiment = "positive" if sentiment_roll < 0.33 else ("neutral" if sentiment_roll < 0.66 else "negative")

            text = generate_text(sentiment, min_words=50, max_words=500)
            texts.append({
                "text_id": str(uuid.uuid4()),
                "student_id": student_id,
                "text": text,
                "sentiment": sentiment,
                "word_count": len(text.split()),
            })

    df_texts = pd.DataFrame(texts)

    logs = []
    for _, row in df_students.iterrows():
        student_id = row["student_id"]
        risk = row["risk_score"]
        n_events = random.randint(50, 500)
        logs.extend(generate_behavioral_logs(student_id, n_events, risk))

    df_logs = pd.DataFrame(logs)

    df_students = df_students.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    n_train = int(N_STUDENTS * TRAIN_RATIO)
    n_val = int(N_STUDENTS * VAL_RATIO)

    train_ids = set(df_students.iloc[:n_train]["student_id"])
    val_ids = set(df_students.iloc[n_train:n_train + n_val]["student_id"])
    test_ids = set(df_students.iloc[n_train + n_val:]["student_id"])

    df_students["split"] = df_students["student_id"].apply(
        lambda x: "train" if x in train_ids else ("val" if x in val_ids else "test")
    )

    students_path = os.path.join(RAW_DIR, "students.csv")
    texts_path = os.path.join(RAW_DIR, "texts.csv")
    logs_path = os.path.join(RAW_DIR, "behavioral_logs.csv")

    df_students_save = df_students.copy()
    df_students_save["assignment_scores"] = df_students_save["assignment_scores"].apply(
        lambda scores: "|".join(f"{s:.1f}" for s in scores)
    )

    df_students_save.to_csv(students_path, index=False)
    df_texts.to_csv(texts_path, index=False)
    df_logs.to_csv(logs_path, index=False)

    for split in ["train", "val", "test"]:
        split_ids = df_students[df_students["split"] == split]["student_id"]
        df_students_save[df_students_save["student_id"].isin(split_ids)].to_csv(
            os.path.join(PROCESSED_DIR, f"students_{split}.csv"), index=False
        )
        df_texts[df_texts["student_id"].isin(split_ids)].to_csv(
            os.path.join(PROCESSED_DIR, f"texts_{split}.csv"), index=False
        )
        df_logs[df_logs["student_id"].isin(split_ids)].to_csv(
            os.path.join(PROCESSED_DIR, f"behavioral_logs_{split}.csv"), index=False
        )

    print("\nDataset generation complete!")
    print(f"  Students:     {len(df_students)} rows")
    print(f"  Texts:        {len(df_texts)} rows")
    print(f"  Behavioral:   {len(df_logs)} rows")
    print(f"\nSplit sizes:")
    for split in ["train", "val", "test"]:
        count = (df_students["split"] == split).sum()
        at_risk_pct = df_students[df_students["split"] == split]["at_risk"].mean() * 100
        print(f"  {split:5s}: {count:4d} students ({at_risk_pct:.1f}% at-risk)")

    total_size = 0
    for path in [students_path, texts_path, logs_path]:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        total_size += size_mb
        print(f"  {os.path.basename(path)}: {size_mb:.2f} MB")
    print(f"  Total raw size: {total_size:.2f} MB")

    evidence_dir = os.path.join(os.path.dirname(__file__), "..", ".sisyphus", "evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    with open(os.path.join(evidence_dir, "task-7-dataset-sizes.txt"), "w") as f:
        f.write("Task 7: Synthetic Dataset Generation\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Students:     {len(df_students)} rows\n")
        f.write(f"Texts:        {len(df_texts)} rows\n")
        f.write(f"Behavioral:   {len(df_logs)} rows\n\n")
        f.write("Split sizes:\n")
        for split in ["train", "val", "test"]:
            count = (df_students["split"] == split).sum()
            at_risk_pct = df_students[df_students["split"] == split]["at_risk"].mean() * 100
            f.write(f"  {split:5s}: {count:4d} students ({at_risk_pct:.1f}% at-risk)\n")
        f.write(f"\nAt-risk distribution: {df_students['at_risk'].mean()*100:.1f}%\n")
        f.write(f"Total raw size: {total_size:.2f} MB\n")

    return df_students, df_texts, df_logs


if __name__ == "__main__":
    generate_dataset()
