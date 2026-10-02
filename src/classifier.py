from __future__ import annotations

from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.nlp import load_role_skills


def _synthetic_corpus() -> tuple[list[str], list[str]]:
    role_skills = load_role_skills()
    docs, labels = [], []
    extras = [
        "internship project github",
        "bachelor of technology computer science",
        "developed application using",
        "worked in team agile",
    ]
    for role, skills in role_skills.items():
        base = " ".join(skills)
        docs.append(f"{role} {base} {extras[0]}")
        docs.append(f"{base} {extras[1]} {extras[2]}")
        docs.append(f"seeking {role} role {base} {extras[3]}")
        labels.extend([role, role, role])
    return docs, labels


@lru_cache(maxsize=1)
def _model() -> Pipeline:
    docs, labels = _synthetic_corpus()
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", MultinomialNB()),
        ]
    )
    pipe.fit(docs, labels)
    return pipe


def suggest_roles(text: str, top_k: int = 3) -> list[tuple[str, float]]:
    model = _model()
    proba = model.predict_proba([text[:8000]])[0]
    labels = model.classes_
    ranked = sorted(zip(labels, proba), key=lambda x: x[1], reverse=True)
    return [(str(role), float(score)) for role, score in ranked[:top_k]]
