<!--
# LLM readability
The 2025 PubMed baseline was obtained before. The relevant files are from the [`pubmed-retina`](https://github.com/ritagonmar/pubmed-retina) repo, specifically from `scripts/`:
- `01-process-baseline.ipynb`
- `02-generate-2025-tsne-plots.ipynb`
- `obtain_BERT_embeddings.py`

 and from `src/pubmed_retina`:
- `embeddings_pubmed_utils.py`
- `process_pubmed_utils.py`
-->

# LLM readability

Quantifying how LLM-assisted writing affects the biomedical literature through changes in readability metrics (e.g., word length). Analysis done in the PubMed data over time (2010-2025).

The PubMed baseline for 2025 was obtained previously in the [`pubmed-retina`](https://github.com/ritagonmar/pubmed-retina) repo; see `scripts/01-process-baseline.ipynb`, `scripts/02-generate-2025-tsne-plots.ipynb`, `scripts/obtain_BERT_embeddings.py`, `src/pubmed_retina/embeddings_pubmed_utils.py`, and `src/pubmed_retina/process_pubmed_utils.py`.

## Pipeline

1. `scripts/01-obtain-yearly-word-counts.py`: loads the PubMed baseline (2010-2025), cleans up abstracts (removes corrections/errata/retractions and boilerplate text), vectorizes abstracts into per-year word counts, and saves `results/variables/yearly-counts.parquet.gzip`.
2. `scripts/02-obtain-word-len.py`: computes weighted word-length statistics (mean, median, std, percentiles, length-category counts) per year from the yearly word counts, saving `results/variables/results_word_len.pkl`.
3. `scripts/03-analysis-word-lenght.ipynb`: loads the word-length statistics and produces plots of mean/median word length over time.
4. `scripts/04-obtain-word-length-exclude-2-3-char.py`: repeats the word-length analysis while excluding words of 2-3 characters, saving results to `results/variables/exclude_2_3_char/`.

Core code used across files lives in `src/llm_readability/word_len.py`.

## Data

Scripts expect the processed PubMed baseline at `/gpfs01/berens/data/data/pubmed_processed` (not included in this repo). Intermediate and final results are written to `results/variables/` and figures to `results/figures/`.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management (Python 3.12).

```bash
uv sync
make install_hooks  # installs pre-commit hooks
```
