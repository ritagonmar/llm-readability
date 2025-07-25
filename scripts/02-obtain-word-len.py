import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import timedelta
from llm_readability.word_len import compute_word_length_statistics

# define paths
variables_path = Path("../results/variables")
assert variables_path.exists(), "The path does not exist"

# compute word length statisctics
## load yearly counts df
yearly_counts_df = pd.read_parquet(
    variables_path / "yearly-counts.parquet.gzip",
    engine="pyarrow",
)
## obtain analysis
start = time.time()
years = np.arange(2010, 2026)
results = compute_word_length_statistics(
    yearly_counts_df, saving_path=variables_path, years=years
)
end = time.time()
runtime_total = end - start
print("Word length statistics runtime: ", str(timedelta(seconds=runtime_total)))
