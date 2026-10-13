import csv
import json
import re
from pathlib import Path

import pandas as pd

# Project root: two levels above this script
ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "raw" /  "cmu_books" / "booksummaries.txt"
OUTPUT_FILE = ROOT / "data" / "processed" / "books.parquet"

COLUMNS = [
    "wiki_id",
    "freebase_id",
    "title",
    "author",
    "publication_date",
    "genres",
    "summary",
]


def clean_text(value):
    """Normalize whitespace and handle missing text."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_genres(value):
    """Convert the dataset's JSON genre mapping to a list."""
    value = clean_text(value)

    if not value:
        return []

    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []

    if isinstance(parsed, dict):
        genres = parsed.values()
    elif isinstance(parsed, list):
        genres = parsed
    else:
        return []

    return sorted({
        clean_text(genre)
        for genre in genres
        if clean_text(genre)
    })


def normalize_key(value):
    """Build a case-insensitive key for deduplication."""
    return clean_text(value).casefold()


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    # Parse the seven tab-separated fields.
    df = pd.read_csv(
        INPUT_FILE,
        sep="\t",
        header=None,
        names=COLUMNS,
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
        on_bad_lines="warn",
    )

    print(f"Rows loaded: {len(df):,}")

    # Clean text fields.
    for col in [
        "wiki_id",
        "freebase_id",
        "title",
        "author",
        "publication_date",
        "summary",
    ]:
        df[col] = df[col].map(clean_text)

    # Parse genres, including empty or malformed JSON.
    df["genres"] = df["genres"].map(parse_genres)

    # Titles are required; authors, genres and summaries may be missing.
    before = len(df)
    df = df[df["title"] != ""].copy()
    print(f"Rows removed for missing titles: {before - len(df):,}")

    # Dedupe using normalized title + author.
    df["_title_key"] = df["title"].map(normalize_key)
    df["_author_key"] = df["author"].map(normalize_key)

    before = len(df)
    df = df.drop_duplicates(
        subset=["_title_key", "_author_key"],
        keep="first",
    ).copy()
    print(f"Duplicate rows removed: {before - len(df):,}")

    df = df.drop(columns=["_title_key", "_author_key"])

    # Build text that can later be sent to an embedding model.
    def make_embedding_text(row):
        parts = [f"Title: {row['title']}"]

        if row["author"]:
            parts.append(f"Author: {row['author']}")

        if row["publication_date"]:
            parts.append(
                f"Publication date: {row['publication_date']}"
            )

        if row["genres"]:
            parts.append(
                "Genres: " + ", ".join(row["genres"])
            )

        if row["summary"]:
            parts.append(f"Summary: {row['summary']}")

        return "\n".join(parts)

    df["embedding_text"] = df.apply(
        make_embedding_text,
        axis=1,
    )

    # Put the processed dataset in its own directory.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(OUTPUT_FILE, index=False)

    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Final rows: {len(df):,}")
    print(f"Final columns: {df.columns.tolist()}")
    print(f"Missing summaries: {(df['summary'] == '').sum():,}")
    print(f"Missing authors: {(df['author'] == '').sum():,}")
    print(f"Missing genres: {df['genres'].map(len).eq(0).sum():,}")

    # Read the file back to verify the output.
    check = pd.read_parquet(OUTPUT_FILE)
    assert len(check) == len(df)
    assert check["embedding_text"].notna().all()

    print("Parquet read-back validation passed.")


if __name__ == "__main__":
    main()