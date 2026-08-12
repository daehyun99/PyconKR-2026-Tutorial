"""
Use this when you can’t connect to the internet.
"""
import pandas as pd
from pgmpy.datasets import load_dataset

def load_college_plans():
    try:
        dataset = load_dataset("college_plans")
        df = dataset.data
    except:
        df = pd.read_csv("../data/college-plans.discrete.txt", sep="\t")   
    data = df.copy()

    for col in ["sex", "ses", "iq", "pe", "cp"]:
        data[col] = pd.to_numeric(
            data[col].astype(str),
            errors="coerce"
        )
    data["cp"] = data["cp"].map({
        1: 2,
        2: 1,
    })
    return data
