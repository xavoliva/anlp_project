import sys
import pandas as pd
from utils import get_files_from_folder
from constants import COLUMNS

files = get_files_from_folder(f"data/{sys.argv[1]}")

if __name__ == "__main__":
    for f in files:
        print(f"{f[:-4]}")
        try:
            df = pd.read_csv(f,
                            usecols=COLUMNS,
                            dtype={
                                "subreddit": "string[pyarrow]",
                                "author": "string[pyarrow]",
                                "body": "string[pyarrow]",
                            })
            df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s').dt.date
            df.to_parquet(f"{f[:-4]}.parquet")
            del df
        except OSError as err:
            print(err)

