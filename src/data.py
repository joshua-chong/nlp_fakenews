from pathlib import Path
import pandas as pd
import re

def load_kaggle_fake_real(data_dir="data"):
    """Load both Kaggle Fake and Real News (True.csv, Fake.csv) and return a clean, shuffled dataset."""
    data_dir = Path(data_dir)
    true_fp = data_dir / "True.csv"
    fake_fp = data_dir / "Fake.csv"

    if not true_fp.exists() or not fake_fp.exists():
        raise FileNotFoundError(f"True.csv and/or Fake.csv not found in {data_dir}")

    true_df = pd.read_csv(true_fp)
    fake_df = pd.read_csv(fake_fp)

    # Label as real (1) and fake (0)
    true_df["label"] = 1
    fake_df["label"] = 0

    # Combine text + title
    true_df["text"] = true_df["text"].fillna("") + " " + true_df["title"].fillna("")
    fake_df["text"] = fake_df["text"].fillna("") + " " + fake_df["title"].fillna("")

    # Use the correct column name: "text"
    df = pd.concat([true_df[["text","label"]], fake_df[["text","label"]]], ignore_index=True)

    # Basic cleaning
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"].str.len() > 20].dropna(subset=["text","label"]).reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


#removing emails, urls and numbers to prevent biasness in model
def simple_clean(s: str) -> str:
    """Light normalization to reduce spurious signals, preserving meaning for bag-of-words models."""
    s = re.sub(r"http\S+", "<URL>", s)
    s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "<EMAIL>", s)
    s = re.sub(r"\d+(?:\.\d+)?", "<NUM>", s)
    return s
