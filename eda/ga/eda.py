import os
from pathlib import Path
import sqlite3

import pandas as pd

data_dir = Path.cwd() / "data"
wzdx_data_file = data_dir / "wzdx.db"
inrix_data_dir = data_dir/ "INRIX data"
inrix_data_1 = inrix_data_dir / "I10-and-I17-1year" / "I10-and-I17-1year.csv"
inrix_data_2 = inrix_data_dir / "Loop101-1year" / "Loop101-1year.csv"
inrix_data_3 = inrix_data_dir / "SR60-1year" / "SR60-1year.csv"

def data_wzdx():
    con = sqlite3.connect(wzdx_data_file)
    cur = con.cursor()

    # The name of table in wzdx is events

    cur.execute("SELECT * FROM events")
    columns = [desc[0] for desc in cur.description]

    data_df = pd.DataFrame(cur.fetchall(), columns=columns)

    con.close()

    return data_df

data = pd.read_csv(inrix_data_2,nrows=1000)
print(data.sample(5))