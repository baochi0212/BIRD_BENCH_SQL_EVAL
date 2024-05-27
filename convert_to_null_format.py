import argparse
import json
import sqlite3
import os
from tqdm import tqdm

def execute_sql(cursor, sql):
    cursor.execute(sql)

    return cursor.fetchall()

def add_nullable(dataset, db_root_path):
    for sample in tqdm(dataset, ncols=0):
        db_name = sample["db_id"]
        conn = sqlite3.connect(os.path.join(db_root_path, db_name, f"{db_name}.sqlite"))
        cursor = conn.cursor()

        for table in sample["schema"]["schema_items"]:
            table_name = table["table_name"]
            results = execute_sql(cursor, "SELECT `notnull` FROM PRAGMA_TABLE_INFO('{}')".format(table_name))
            notnull_indicators = [result[0] for result in results]
            table["notnull_indicators"] = notnull_indicators
    return dataset

def main(args):
    original_dataset = json.load(open(args.dataset))
    new_dataset = add_nullable(original_dataset, args.db_root_path)

    with open(args.outfile, "w") as f:
        json.dump(new_dataset, f, indent="\t", ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, help="path to original dataset")
    parser.add_argument("--db_root_path", type=str, help="path to the root databases")
    parser.add_argument("--outfile", type=str, help="result file path")

    args = parser.parse_args()
    main(args)
