#!/usr/bin/env python3
import sys
import pandas as pd

def fix_file(infile: str, outfile: str):
    df = pd.read_csv(infile)

    # Case A: separate date/time columns
    if {'date','time'}.issubset(df.columns):
        df['datetime'] = pd.to_datetime(
            df['date'].astype(str) + ' ' + df['time'].astype(str),
            errors='coerce', infer_datetime_format=True
        )
        df.drop(columns=['date','time'], inplace=True)

    # Case B: single datetime column
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(
            df['datetime'], errors='coerce', infer_datetime_format=True
        )
    else:
        raise ValueError(
            "Input CSV must have either columns ['date','time'] or 'datetime'"
        )

    if df['datetime'].isna().any():
        n_bad = df['datetime'].isna().sum()
        raise ValueError(f"{n_bad} rows failed to parse into datetime")

    # place datetime at the end
    cols = [c for c in df.columns if c != 'datetime'] + ['datetime']
    df = df[cols]

    df.to_csv(outfile, index=False)
    print(f"✔ Wrote cleaned data to {outfile}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_datetime.py <input.csv> <output.csv>")
        sys.exit(1)
    fix_file(sys.argv[1], sys.argv[2])
