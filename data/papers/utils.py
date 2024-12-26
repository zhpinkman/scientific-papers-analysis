import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# load the data using the pathlib library and relative to the current directory
# the data is in the data folder

from pathlib import Path

data_file = Path(__file__).parent / "arxiv-metadata-oai-snapshot.json"


def load_data():
    with open(data_file, "r") as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)


if __name__ == "__main__":
    df = load_data()
    from IPython import embed

    embed()
