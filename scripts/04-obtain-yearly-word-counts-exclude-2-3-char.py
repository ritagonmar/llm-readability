import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import timedelta
from llm_readability.word_len import *


# define paths
variables_path = Path("../results/variables")
print(variables_path.resolve())
berenslab_data_path = Path("/gpfs01/berens/data/data/pubmed_processed")
assert variables_path.exists(), "The path does not exist"
assert berenslab_data_path.exists(), "The path does not exist"

saving_path = variables_path / Path("exclude_2_3_char")
(saving_path).mkdir(parents=True, exist_ok=True)

# load data
start = time.time()
df = load_data(start_year=2010, end_year=2025)
end = time.time()
runtime_total = end - start
print("Loading data runtime: ", str(timedelta(seconds=runtime_total)))

# clean up abstracts
start = time.time()
cleanup_abstracts_inplace(df)
end = time.time()
runtime_total = end - start
print("Cleaning abstracts runtime: ", str(timedelta(seconds=runtime_total)))

# compute yearly counts
start = time.time()
years = np.arange(2010, 2026)
yearly_counts_df = vectorize_abstracts(
    df, years, saving_path, token_pattern=r"\b[a-zA-Z]{4,}\b"
)
print(yearly_counts_df.columns)
print(type(yearly_counts_df.columns[0]))
end = time.time()
runtime_total = end - start
print("Yearly counts runtime: ", str(timedelta(seconds=runtime_total)))

# compute word length statisctics
## load yearly counts df
# yearly_counts_df = pd.read_parquet(
#     saving_path / "yearly-counts.parquet.gzip",
#     engine="pyarrow",
# )
start = time.time()
results = compute_word_length_statistics(
    yearly_counts_df, saving_path=saving_path, years=years
)
end = time.time()
runtime_total = end - start
print("Word length statistics runtime: ", str(timedelta(seconds=runtime_total)))
