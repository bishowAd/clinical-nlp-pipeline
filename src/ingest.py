import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db import get_engine

def load_raw_data():
    print("Reading CSV file...")
    df = pd.read_csv("data/mtsamples.csv")

    print(f"Found {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")

    df = df[["description", "medical_specialty", "sample_name", "transcription", "keywords"]]

    df.columns = ["description", "medical_specialty", "title", "transcription", "keywords"]

    before = len(df)
    df = df.dropna(subset=["transcription", "medical_specialty"])
    after = len(df)
    print(f"Dropped {before - after} rows with missing data")

    df["medical_specialty"] = df["medical_specialty"].str.strip()

    engine = get_engine()
    df.to_sql("raw_notes", con=engine, if_exists="replace", index=True, index_label="id")
    print(f"Successfully loaded {len(df)} rows into raw_notes table")

if __name__ == "__main__":
    load_raw_data()