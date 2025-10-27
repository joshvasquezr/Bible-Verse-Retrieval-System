import os
import re
from typing import List, Tuple

import nltk
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd
from collections import Counter
from pathlib import Path


THEMES: List[str] = [
    "Love", "Joy", "Peace", "Grief", "Faith",
    "Fear", "Mercy", "Grace", "Doubt", "Forgiveness",
]

# stopwords from KJV (if available)
KJV_STOPWORDS: List[str] = []


def simple_preprocess(text: str) -> str:
    """Lowercase, remove punctuation/numbers, collapse whitespace.
    Keeps it simple to avoid heavy NLTK downloads in containers.
    """
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)  
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_training_data() -> Tuple[List[str], List[str]]:
    """Load labeled verse→theme pairs.
    Priority:
      1) CSV file named verses_themes.csv located under ./api or working dir
         with columns: verse, theme
      2) Built-in tiny seed dataset as a fallback (for first run).
    """
    possible_paths = [
        os.path.join(os.getcwd(), "verses_themes.csv"),
        os.path.join(os.path.dirname(__file__), "verses_themes.csv"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if {"verse", "theme"}.issubset(df.columns):
                verses = df["verse"].astype(str).tolist()
                labels = df["theme"].astype(str).tolist()
                return verses, labels

    # small seed examples per theme
    seed: List[Tuple[str, str]] = [
        ("Love is patient, love is kind.", "Love"),
        ("We love because he first loved us.", "Love"),
        ("The joy of the Lord is your strength.", "Joy"),
        ("Rejoice in the Lord always.", "Joy"),
        ("Blessed are the peacemakers.", "Peace"),
        ("Peace I leave with you; my peace I give you.", "Peace"),
        ("Jesus wept.", "Grief"),
        ("My soul is in deep anguish.", "Grief"),
        ("For we walk by faith, not by sight.", "Faith"),
        ("Now faith is confidence in what we hope for.", "Faith"),
        ("Do not fear, for I am with you.", "Fear"),
        ("Perfect love drives out fear.", "Fear"),
        ("Blessed are the merciful.", "Mercy"),
        ("His mercies are new every morning.", "Mercy"),
        ("By grace you have been saved.", "Grace"),
        ("My grace is sufficient for you.", "Grace"),
        ("Why did you doubt?", "Doubt"),
        ("Help my unbelief.", "Doubt"),
        ("Forgive us our debts, as we forgive our debtors.", "Forgiveness"),
        ("Forgive, and you will be forgiven.", "Forgiveness"),
    ]
    verses, labels = zip(*seed)
    return list(verses), list(labels)


def load_kjv_stopwords(top_k: int = 150) -> List[str]:
    """Derive common/archaic tokens from data/kjv.txt to use as light stopwords.
    Falls back to no extra stopwords if the file is not present.
    """
    candidates = [
        Path(os.getcwd()) / "data" / "kjv.txt",
        Path(__file__).resolve().parents[1] / "data" / "kjv.txt",
    ]
    kjv_path = next((p for p in candidates if p.exists()), None)
    if not kjv_path:
        return []

    counter: Counter[str] = Counter()
    try:
        with kjv_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                toks = simple_preprocess(line).split()
                counter.update([t for t in toks if len(t) > 2])
    except Exception:
        return []

    always_keep = {
        "god", "lord", "jesus", "christ", "spirit", "faith",
        "love", "grace", "mercy", "sin", "forgive", "forgiveness",
        "fear", "peace", "joy", "doubt",
    }
    archaic = {"thee", "thou", "thy", "thine", "ye", "art", "hath", "doth", "unto", "saith", "verily"}

    most_common = [w for w, _ in counter.most_common(top_k * 3)]
    derived = [w for w in most_common if w not in always_keep]
    stopset = list(dict.fromkeys([*archaic, *derived]))[:top_k]
    return stopset


def build_pipeline() -> Pipeline:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            pass

    global KJV_STOPWORDS
    KJV_STOPWORDS = load_kjv_stopwords(top_k=150)

    vectorizer = TfidfVectorizer(
        preprocessor=simple_preprocess,
        ngram_range=(1, 2),
        min_df=1,
        stop_words=KJV_STOPWORDS if KJV_STOPWORDS else None,
    )
    clf = MultinomialNB()
    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf),
    ])


def train_model() -> Pipeline:
    X, y = load_training_data()
    pipe = build_pipeline()
    pipe.fit(X, y)
    return pipe


# init app and model
app = FastAPI(title="Bible Verse Theme Classifier", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL: Pipeline = train_model()


class PredictRequest(BaseModel):
    verse: str


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": MODEL is not None, "kjv_stopwords": len(KJV_STOPWORDS)}


@app.post("/predict")
async def predict(req: PredictRequest):
    text = req.verse or ""
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="'verse' must be a non-empty string")

    probs = None
    try:
        probs = MODEL.predict_proba([text])[0]
        labels = MODEL.named_steps["clf"].classes_
        # Find best
        best_idx = probs.argmax()
        theme = str(labels[best_idx])
        confidence = float(probs[best_idx])
    except AttributeError:

        pred = MODEL.predict([text])[0]
        theme = str(pred)
        confidence = 1.0

    return {"theme": theme, "confidence": round(confidence, 4)}


@app.get("/")
async def root():
    return {"message": "Bible Verse Retrieval API - Milestone 1", "themes": THEMES}
