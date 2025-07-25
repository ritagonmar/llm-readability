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
yearly_counts_df = vectorize_abstracts(df, years, variables_path)
end = time.time()
runtime_total = end - start
print("Yearly counts runtime: ", str(timedelta(seconds=runtime_total)))
