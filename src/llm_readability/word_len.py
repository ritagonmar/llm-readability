# modified from repo llm-excess-vocab/scripts/02_preprocess_and_count_utils.py

import pandas as pd
import numpy as np
import pickle
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt
from tqdm import tqdm

# TODO: maybe move to scripts, because this currently runs an analysis


def load_data(start_year=2010, end_year=2024):
    print("Loading data...", flush=True)

    loading_path = Path("/gpfs01/berens/data/data/pubmed_processed")

    df = pd.read_parquet(
        loading_path / "pubmed_baseline_2025.parquet.gzip",
        engine="pyarrow",
    )

    if end_year == 2025:
        df2 = pd.read_parquet(
            loading_path / "pubmed_daily_updates_2025_v1.parquet.gzip",
            engine="pyarrow",
        )

        df = pd.concat((df, df2))
        df = df.groupby(["PMID"]).last()

    print(f"Found {len(df)} papers.", flush=True)

    df = df[(df.Year >= start_year) & (df.Year <= end_year)]
    print(
        f"Kept {len(df)} papers from {start_year}--{end_year}.", flush=True
    )  # 15103887

    return df


def cleanup_abstracts_inplace(df):
    # Titles filter

    ind = df.Title.str.contains("Correction:", regex=False)
    ind |= df.Title.str.contains("Correction to:", regex=False)
    ind |= df.Title.str.contains("Erratum:", regex=False)
    ind |= df.Title.str.contains("Erratum to:", regex=False)
    ind |= df.Title.str.contains("Corrigendum:", regex=False)
    ind |= df.Title.str.contains("Corrigendum to:", regex=False)
    ind |= df.Title.str.contains("Retracted:", regex=False)
    ind |= df.Title == "Retraction"

    df.loc[ind, "AbstractText"] = ""

    print(
        f"Removing {np.sum(ind)} abstracts (corrections/errata/retractions/etc).\n",
        flush=True,
    )  # 3514

    # Abstracts filter

    to_replace = {
        "&ldquo;": {"&ldquo;": '"', "&rdquo;": '"'},
        "&lsquo;": {"&lsquo;": "'", "&rsquo;": "'"},
        "&nbsp;": {"&nbsp;": " "},
        "&shy;": {"&shy;": ""},
        "&mdash;": {"&mdash;": "---"},
        "&ndash;": {"&ndash;": "--"},
        "u2002": {"\\u2002": " "},
        "<p>": {"<p>": "", "</p>": ""},
        "<em>": {"<em>": "", "</em>": ""},
        "This article": {"^This article has been.*": ""},
        "This manuscript": {"^This manuscript has been.*": ""},
        "The above article": {
            "^The above article.*": "",
            ".*The above article, published online.*": "",
        },
        "http": {"^http.*": ""},
        "For complete details": {
            "\s*For complete details on the use and execution of this protocol.*": "",
        },
        "For further information": {
            "For further information please consult linked data\.*": "",
        },
        "Communicated by": {"\s*\(?Communicated by.{0,100}$": ""},
        "Graphical abstract": {"\.\s*Graphical abstract.*": "."},
        "GRAPHICAL ABSTRACT": {"\.\s*GRAPHICAL ABSTRACT.*": "."},
        "VIDEO ABSTRACT": {"\.\s*VIDEO ABSTRACT.*": "."},
        "Video Abstract": {"\s*Video Abstract Available\.*": ""},
        "MINI ABSTRACT": {"\.\s*MINI ABSTRACT.*": "."},
        "ABSTRACT": {
            "^ABSTRACT[:.]?\s*": "",
            "^Abstract ABSTRACT[:.]?\s*": "",
            "^.{0,200} ABSTRACT: ": "",
        },
        "Abstract": {"^Abstract:?\s*": ""},
        "CONSPECTUS": {"CONSPECTUS: ": ""},
        "THIS ARTICLE": {
            "\s*THIS ARTICLE HAD BEEN MADE AVAILABLE FREE OF CHARGE.*": ""
        },
        "Copyright ©": {"\s*Copyright ©.*": ""},
        " © ": {
            "\.\s*[^.]*[0-9]\. © [12].*": ".",
            "\. [a-zA-Z]+\.$": ".",
            "\. [ab-zA-Z]+\.$": ".",
            "\. [abc-zA-Z]+\.$": ".",
            "\. [abcd-zA-Z]+\.$": ".",
            "\. Pediatr Pulmonol.$": ".",
            "\. Lasers Surg.$": ".",
            "\s+© .*": "",
        },
        ".© ": {
            "\.\s*[^.]*[0-9]\.© [12].*": ".",
            "\. [a-zA-Z]+\.$": ".",
            "\. [ab-zA-Z]+\.$": ".",
            "\. [abc-zA-Z]+\.$": ".",
            "\. [abcd-zA-Z]+\.$": ".",
            "\. Pediatr Pulmonol.$": ".",
            "\. Lasers Surg.$": ".",
            "\s+\.© .*": "",
        },
        " ©20": {
            "\.\s*[^.]*[0-9]\. ©20.{0,20}$": ".",
            "\s+©20.{0,20}$": "",
            "\.\s*[^.]*[0-9]\. ©20[0-2][0-9] AACRSee.*": ".",
            "\s+©20[0-2][0-9] AACRSee.*": "",
        },
        "Wiley Periodicals Inc": {
            "\.\s*[^.]*[0-9]\.\s*Published [12][890][0-9][0-9] Wiley Periodicals Inc.*": ".",
            "\. [a-zA-Z]+\.$": ".",
            "\. [ab-zA-Z]+\.$": ".",
            "\. [abc-zA-Z]+\.$": ".",
            "\. [abcd-zA-Z]+\.$": ".",
            "\s*Published [12][890][0-9][0-9] Wiley Periodicals Inc.*": ".",
        },
        "doi": {
            "\s*doi:\s*10\.[0-9a-zA-Z\.\/\-]*\s*$": "",
            "\s*doi:\s*10\.[0-9a-zA-Z\.\/\-]*\s*\(.*\)\.?\s*$": "",
            "\s*http://dx\.doi\.org/10\.[0-9a-zA-Z\.\/\-]*\s*$": "",
        },
        "DOI": {"\s*DOI: http://dx.doi.org/[0-9a-zA-Z\.\/\-]*\s*$": ""},
        "PMID": {".*PubMed PMID: [0-9]*\.\s*": ""},
        "Epub": {"\sEpub.{0,10}[12][890][0-9][0-9]\.?\s*$": ""},
        "Level of Evidence": {"\s*Level of Evidence:?\s*[0-9IV].*": ""},
        "LEVEL OF EVIDENCE": {"\s*LEVEL OF EVIDENCE:?\s*[0-9IV].*": ""},
        "Technical Efficacy": {"\s*[0-9] Technical Efficacy: Stage [0-9].*": ""},
        "Geriatr": {
            "\s*Geriatr Gerontol Int [\s0-9,;:\-\.\(\)]*$": "",
            "\s*J Am Geriatr Soc [\s0-9,;:\-\.\(\)]*$": "",
        },
        "Genet Med": {"\s*Genet Med [\s0-9,;:\-\.\(\)]*$": ""},
        "Ann Neurol": {
            # Sometimes occurs twice
            "\s*Ann Neurol [\s0-9,;:\-\.\(\)]*$": "",
            "\s*Ann Neurol [\s01-9,;:\-\.\(\)]*$": "",
        },
        "ANN NEUROL": {"\s*ANN NEUROL [\s0-9,;:\-\.\(\)]*$": ""},
        "J Drugs Dermatol": {"\s*J Drugs Dermatol\. [\s0-9,;:\-\.\(\)]*$": ""},
        "Infect Control Hosp": {
            "\s*Infect Control Hosp Epidemiol [\s0-9,;:\-\.\(\)]*$": ""
        },
        "Magn. Reson.": {
            "\.\s*[0-9]\s*J\. Magn\. Reson\. Imaging [\s0-9,;:\-\.\(\)]*$": ""
        },
        "MAGN. RESON.": {
            "\.\s*[0-9]\s*J\. MAGN\. RESON\. IMAGING [\s0-9,;:\-\.\(\)]*$": ""
        },
        "Magnetic Resonance": {
            "\s*Magnetic Resonance in Medicine published by Wiley Periodicals\.*": "",
        },
        "(Pediatr Dent": {"\s*\(Pediatr Dent 20.*": ""},
        "Environ Toxicol Chem": {"\s*Environ Toxicol Chem [\s0-9,;:\-\.\(\)]*$": ""},
        "Environ Health Perspect": {
            "\s*Environ Health Perspect [\s0-9,;:\-\.\(\)]*$": ""
        },
        "Antioxid. Redox Signal.": {
            "\s*Antioxid\. Redox Signal\. [\s0-9,;:\-\.\(\)]*$": ""
        },
        "J Orthop Sports Phys Ther": {
            "\s*J Orthop Sports Phys Ther\.? [\sA0-9,;:\-\.\(\)]*$": ""
        },
        "J Strength Cond Res": {
            ".*J Strength Cond Res.{0,20}20[012][0-9]-([A-Z])": "\\1"
        },
        "Turk J Pediatr": {".*Turk J Pediatr [\s0-9,;:\-\.\(\)]*([A-Z])": "\\1"},
        "Laryngoscope": {
            "\s*[1-9][a-zA-Z]?\. Laryngoscope[^\.]*\.\s*$": "",
            "[\.][^\.]*Laryngoscope[^\.]*\.\s*$": ".",
            "\s*N/A\.$": "",
        },
        "Indian J Crit Care Med": {
            # Author list. Title. Indian J Crit Care Med
            "\s*[^\.]*\.[^\.]*[\.?!] Indian J Crit Care Med [\s0-9,;:\-\.\(\)]*$": "",
        },
        "Int J Clin Pediatr Dent": {
            # Author list. Title. Int J Clin Pediatr Dent
            "\s*[^\.]*\.[^\.]*[\.?!] Int J Clin Pediatr Dent [\s0-9,;:\-\.\(\)]*$": "",
        },
        "J Clin Sleep Med": {
            # Author list. Title. Int J Clin Pediatr Dent
            "\s*[^\.]*\.[^\.]*[\.?!] J Clin Sleep Med\. [\s0-9,;:\-\.\(\)]*$": "",
        },
        "Hepatology": {
            "\(Hepatology [\s0-9,;:\-\.\(\)]*\)\.?\s*$": "",
            "\(Hepatology Communications [\s0-9,;:\-\.\(\)]*\)\.?\s*$": "",
        },
        "AASLD": {"[^\.]*AASLD\.\s*$": ""},
        "Database Record": {
            "\s*\(PsycINFO Database Record.*": "",
            "\s*\(PsycInfo Database Record.*": "",
        },
        "advance online publication": {
            "[^\.]* advance online publication,?\s*[0-9][0-9] [A-Za-z]* [0-9]{4}[\.;,]?\s*$": ""
        },
        "This article is protected": {"\sThis article is protected by copyright.*": ""},
        "This article is part": {
            "\s*This article is part of the themed issue.*": "",
            "\s*This article is part of a themed issue.*": "",
            "\s*This article is part of a themed section.*": "",
            "\s*This article is part of a Special Issue entitled.*": "",
        },
        "Elsevier Ltd": {"\s*20[012][0-9] Elsevier Ltd.*": ""},
        "How to cite this article:": {"\s*How to cite this article:.*": ""},
        "Cite this article:": {"\s*Cite this article:.*": ""},
        "Citation": {"\s+Citation: .*": ""},
        "ClinicalTrials.gov:": {"\s*\(?ClinicalTrials.gov: .{0,100}$": ""},
        ".].": {"\[[^\[]*[0-9]\.\]\.?$": ""},
        "https://youtu.be": {
            "\.\s*[^.]*: https://youtu.be.{0,100}$": ".",
            "\. https://youtu.be.{0,50}.$": ".",
        },
        "The virtual slide(s) for this article": {
            "\s*The virtual slide\(s\) for this article.*": ""
        },
        "IMPACT STATEMENT": {"\.\s*IMPACT STATEMENT[A-Z].*": "."},
        "Impact statement": {"\.\s*Impact Statement[A-Z].*": "."},
        ("RESULTS", "CONCLUSIONS"): {
            "\.\s*PURPOSE[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*BACKGROUND[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*INTRODUCTION[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*OBJECTIVE[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*MATERIALS AND METHODS[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*MATERIALS & METHODS[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*METHODS?[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*METHODOLOGY[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*DESIGN[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*STUDY DESIGN[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*KEY RESULTS[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*RESULTS[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*CONCLUSIONS[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*CONCLUSIONS AND INFERENCES[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*CONCLUSIONS & INFERENCES[.: ][.:]?\s*([A-Z])": ". \\1",
        },
        ("Results", "Conclusions"): {
            "\.\s*Purpose[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Background[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Introduction[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Objective[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Materials and methods[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Materials and Methods[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Materials & Methods[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Methods?[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Methodology[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Design[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Study Design[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Key Results[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Results[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Conclusions[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Conclusions and inferences[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Conclusions and Inferences[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Conclusions & inferences[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Conclusions & Inferences[.: ][.:]?\s*([A-Z])": ". \\1",
        },
        "Expert commentary": {
            "\.\s*Expert commentary:\s*": ". ",
            "\.\s*Areas covered:\s*": ". ",
        },
        "Details of funding": {"\s*Details of funding are provided.*": ""},
        "This journal requires": {"\s*This journal requires.*": ""},
        "Proprietary or commercial disclosure": {
            "\s*Proprietary or commercial disclosure.*": ""
        },
        "See acknowledgments": {".\s*See acknowledgments.\s*$": "."},
        "This article is one of ten": {"\s*This article is one of ten reviews.*": ""},
        "In an effort to expedite the publication of articles": {
            "^In an effort to expedite the publication of articles.*": ""
        },
        "For complete coverage": {
            "\s*For complete coverage of all related areas of Endocrinology.*": ""
        },
        "Abbreviations": {
            "\.\s*Abbreviations:.*": ".",
            "\.\s*Abbreviations [Uu]sed:.*": ".",
        },
        "ABBREVIATIONS": {
            "\.\s*ABBREVIATIONS:.*": ".",
            "\.\s*ABBREVIATIONS USED:.*": ".",
        },
        "Registration number": {"\s*Registration number of the clinical trial:.*": ""},
        " de ": {
            "\.\s*:?\s*[Aa]nalisar .*": ".",
            "\.\s*:?\s*[Dd]escrever .*": ".",
            "\.\s*:?\s*[Ii]mplementar .*": ".",
            "\.\s*:?\s*[Cc]ompreender .*": ".",
            "\.\s*:?\s*[Aa]valiar .*": ".",
            "\.\s*:?\s*[Ee]stimar .*": ".",
            "\.\s*:?\s*[Dd]eterminar .*": ".",
            "\.\s*:?\s*[Rr]ealizar .*": ".",
            "\.\s*:?\s*[Cc]aracterizar .*": ".",
            "\.\s*:?\s*[Ii]dentificar .*": ".",
            "\.\s*:?\s*[Dd]iscutir .*": ".",
            "\.\s*:?\s*[Cc]onhecer .*": ".",
            "\.\s*:?\s*[Cc]onocer .*": ".",
            "\.\s*:?\s*Resumo .*": ".",
        },
    }

    all_affected_abstracts = np.zeros(len(df), dtype=bool)

    for search_string in to_replace:
        if type(search_string) == str:
            print(f"Searching for: {search_string}", end="", flush=True)
            ind = df.AbstractText.str.contains(search_string, regex=False)
        else:
            print("Searching for: " + " + ".join(search_string), end="", flush=True)
            ind = np.ones(len(df), dtype=bool)
            for search_str in search_string:
                ind &= df.AbstractText.str.contains(search_str, regex=False)

        print(f" --> found {np.sum(ind)} abstracts.", flush=True)

        for replace_string in to_replace[search_string]:
            s = (
                df[ind]
                .AbstractText.str.extract("(" + replace_string + ")")
                .values[:, 0]
            )

            ind2 = [type(ss) == str for ss in s.ravel()]
            all_affected_abstracts[np.where(ind)[0][ind2]] = True

            s = [ss[:75] for ss in s.ravel() if type(ss) == str]
            if len(s) > 0:
                print(
                    f"   Found {len(s)} abstract(s) with string(s) to replace:",
                    flush=True,
                )
                print("      " + "\n      ".join(s[:5]) + "\n", flush=True)
            else:
                print("   Found nothing to replace.\n", flush=True)
            df.loc[ind, "AbstractText"] = df[ind].AbstractText.str.replace(
                replace_string, to_replace[search_string][replace_string], regex=True
            )

    print(
        f"In total {np.sum(all_affected_abstracts)} were edited.", flush=True
    )  # 270189


def vectorize_abstracts(df, years, saving_path):
    vectorizer = CountVectorizer(
        lowercase=True,
        token_pattern=r"\b[a-zA-Z]+\b",  # Only alphabetic words
        # max_features=max_features,
        binary=False,
        dtype=np.int64,
        min_df=1e-6,
    )
    X = vectorizer.fit_transform(df.AbstractText.values)  # ~30 min

    print(f"Count matrix computed: {X.shape}", flush=True)  # 14448711 x 4179571

    # extract vocabulary
    vocabulary = vectorizer.vocabulary_

    # save results
    sp.sparse.save_npz(saving_path / "words_counts", X)

    f = open(saving_path / "vocabulary_word_counts.pkl", "wb")
    pickle.dump(vocabulary, f)
    f.close()

    # compute stuff
    words = vectorizer.get_feature_names_out()
    # years = np.arange(2010, 2025)
    counts = np.zeros((words.size, years.size))
    totals = np.zeros(years.size)

    for i, year in enumerate(years):
        ind = df.Year.values == year
        counts[:, i] = np.array(np.sum(X[ind, :], axis=0)).ravel()
        totals[i] = np.sum(ind)

    df = pd.DataFrame(
        dict(zip(["word"] + list(years), [words] + list(counts.astype(int).T)))
    )
    df.loc[len(df)] = [""] + list(totals.astype(int))

    df.to_parquet(
        saving_path / "yearly-counts.parquet.gzip",
        index=False,
        engine="pyarrow",
        compression="gzip",
    )

    return df


############### CLAUDE CODE ###################################################


def compute_word_length_statistics(yearly_counts_df, saving_path, years=None):
    """
    Efficiently compute word length statistics from word counts dataframe.

    Args:
        df: DataFrame with 'word' column and year columns containing counts
        years: Array of years corresponding to the year columns

    Returns:
        Dictionary containing word length statistics and analysis results
    """
    if years is None:
        years = np.array(yearly_counts_df.columns[1:], dtype=int)

    # Remove the totals row (last row with empty word)
    word_counts_df = yearly_counts_df[yearly_counts_df["word"] != ""].copy()

    # Compute word lengths vectorized
    word_lengths = word_counts_df["word"].str.len().values

    # Extract count matrix (all year columns)
    year_columns = [str(year) for year in years]  # they are not str!!
    print(year_columns)
    print(type(year_columns))
    print(type(year_columns[0]))
    print(years)
    print(type(years))
    print(type(years[0]))
    print(word_counts_df.columns)
    print(word_counts_df.columns[-1])
    count_matrix = word_counts_df[year_columns].values

    # Precompute word length categories for efficiency
    length_categories = np.arange(1, word_lengths.max() + 1)

    # Initialize results storage
    results = {
        "years": years,
        "word_lengths": word_lengths,
        "count_matrix": count_matrix,
        "length_distribution": {},
        "mean_length": np.zeros(len(years)),
        "median_length": np.zeros(len(years)),
        "std_length": np.zeros(len(years)),
        "length_percentiles": {},
        "length_category_counts": {},
        "weighted_length_stats": {},
    }

    print("Computing word length statistics...")

    # Compute statistics for each year efficiently
    for i, year in tqdm(enumerate(years)):  # ENH: move tqdm inside enumerate
        year_counts = count_matrix[:, i]

        # # Skip if no words for this year
        # if year_counts.sum() == 0:
        #     continue

        # Compute weighted statistics (weighted by word frequency)
        total_words = year_counts.sum()

        # Mean word length (weighted by frequency)
        weighted_mean = np.average(word_lengths, weights=year_counts)
        results["mean_length"][i] = weighted_mean

        # For median and other percentiles, we need to expand the distribution
        # This is more memory efficient than creating full word list
        expanded_lengths = np.repeat(word_lengths, year_counts.astype(int))

        if len(expanded_lengths) > 0:
            results["median_length"][i] = np.median(expanded_lengths)
            results["std_length"][i] = np.std(expanded_lengths)

            # Compute percentiles
            percentiles = [10, 25, 75, 90, 95, 99]
            year_percentiles = np.percentile(expanded_lengths, percentiles)
            results["length_percentiles"][str(year)] = dict(
                zip(percentiles, year_percentiles)
            )

        # # Count words by length category  --> INEFFICIENT WAY
        # length_counts = np.zeros(len(length_categories))
        # for j, length in enumerate(length_categories):
        #     mask = word_lengths == length
        #     length_counts[j] = year_counts[mask].sum()

        # Count words by length category (vectorized)
        length_counts = np.bincount(
            word_lengths, weights=year_counts
        )  # , minlength=length_categories.max()+1)[1:]

        results["length_category_counts"][str(year)] = {
            "lengths": length_categories,
            "counts": length_counts,
            "proportions": (
                length_counts / total_words if total_words > 0 else length_counts
            ),
        }

    # # Compute overall corpus statistics
    # total_counts = count_matrix.sum(axis=1)
    # corpus_total = total_counts.sum()

    # results['corpus_stats'] = {
    #     'total_words': corpus_total,
    #     'unique_words': len(word_lengths),
    #     'mean_word_length': np.average(word_lengths, weights=total_counts),
    #     'vocabulary_size_by_length': {
    #         length: np.sum(word_lengths == length) for length in length_categories
    #     }
    # }
    f = open(saving_path / "results_word_len.pkl", "wb")
    pickle.dump(results, f)
    f.close()

    return results


def create_word_length_analysis_plots(results, figsize=(15, 12)):
    """
    Create comprehensive visualization of word length analysis.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("Word Length Analysis Over Time", fontsize=16, fontweight="bold")

    years = results["years"]

    # 1. Mean word length over time
    axes[0, 0].plot(years, results["mean_length"], marker="o", linewidth=2)
    axes[0, 0].set_title("Mean Word Length Over Time")
    axes[0, 0].set_xlabel("Year")
    axes[0, 0].set_ylabel("Mean Length (characters)")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Distribution statistics over time
    axes[0, 1].plot(years, results["mean_length"], label="Mean", marker="o")
    axes[0, 1].plot(years, results["median_length"], label="Median", marker="s")
    axes[0, 1].fill_between(
        years,
        results["mean_length"] - results["std_length"],
        results["mean_length"] + results["std_length"],
        alpha=0.3,
        label="±1 STD",
    )
    axes[0, 1].set_title("Word Length Distribution Stats")
    axes[0, 1].set_xlabel("Year")
    axes[0, 1].set_ylabel("Length (characters)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Heatmap of word length proportions over time
    # Create matrix for heatmap
    max_length = max(results["length_category_counts"][str(years[0])]["lengths"])
    heatmap_data = np.zeros(
        (min(20, max_length), len(years))
    )  # Limit to first 20 lengths

    for i, year in enumerate(years):
        year_data = results["length_category_counts"][str(year)]
        lengths = year_data["lengths"][:20]  # First 20 lengths
        proportions = year_data["proportions"][:20]
        heatmap_data[: len(lengths), i] = proportions

    im = axes[0, 2].imshow(heatmap_data, aspect="auto", cmap="viridis")
    axes[0, 2].set_title("Word Length Proportions Heatmap")
    axes[0, 2].set_xlabel("Year")
    axes[0, 2].set_ylabel("Word Length")
    axes[0, 2].set_yticks(range(0, min(20, max_length), 2))
    axes[0, 2].set_yticklabels(range(1, min(21, max_length + 1), 2))
    axes[0, 2].set_xticks(range(0, len(years), 2))
    axes[0, 2].set_xticklabels(years[::2], rotation=45)
    plt.colorbar(im, ax=axes[0, 2])

    # 4. Word length distribution for first and last years
    first_year, last_year = str(years[0]), str(years[-1])
    first_data = results["length_category_counts"][first_year]
    last_data = results["length_category_counts"][last_year]

    axes[1, 0].bar(
        first_data["lengths"][:15],
        first_data["proportions"][:15],
        alpha=0.7,
        label=first_year,
        width=0.4,
    )
    axes[1, 0].bar(
        last_data["lengths"][:15] + 0.4,
        last_data["proportions"][:15],
        alpha=0.7,
        label=last_year,
        width=0.4,
    )
    axes[1, 0].set_title(f"Word Length Distribution: {first_year} vs {last_year}")
    axes[1, 0].set_xlabel("Word Length")
    axes[1, 0].set_ylabel("Proportion")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Percentile evolution
    percentiles_data = results["length_percentiles"]
    p25_values = [percentiles_data[str(year)][25] for year in years]
    p75_values = [percentiles_data[str(year)][75] for year in years]
    p95_values = [percentiles_data[str(year)][95] for year in years]

    axes[1, 1].plot(years, p25_values, label="25th percentile", marker="o")
    axes[1, 1].plot(years, p75_values, label="75th percentile", marker="s")
    axes[1, 1].plot(years, p95_values, label="95th percentile", marker="^")
    axes[1, 1].set_title("Word Length Percentiles Over Time")
    axes[1, 1].set_xlabel("Year")
    axes[1, 1].set_ylabel("Length (characters)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # # 6. Vocabulary size by length category
    # vocab_by_length = results["corpus_stats"]["vocabulary_size_by_length"]
    # lengths = list(vocab_by_length.keys())[:20]  # First 20 lengths
    # counts = [vocab_by_length[length] for length in lengths]

    # axes[1, 2].bar(lengths, counts, alpha=0.7)
    # axes[1, 2].set_title("Vocabulary Size by Word Length")
    # axes[1, 2].set_xlabel("Word Length")
    # axes[1, 2].set_ylabel("Number of Unique Words")
    # axes[1, 2].grid(True, alpha=0.3)

    # plt.tight_layout()
    # return fig


# def generate_word_length_report(results: Dict[str, Any]) -> str:
#     """
#     Generate a comprehensive text report of word length analysis.
#     """
#     corpus_stats = results['corpus_stats']
#     years = results['years']

#     report = f"""
# WORD LENGTH ANALYSIS REPORT
# ===========================

# CORPUS OVERVIEW:
# - Total words analyzed: {corpus_stats['total_words']:,}
# - Unique words in vocabulary: {corpus_stats['unique_words']:,}
# - Overall mean word length: {corpus_stats['mean_word_length']:.2f} characters
# - Time period: {years[0]} - {years[-1]}

# TEMPORAL TRENDS:
# - Mean word length change: {results['mean_length'][-1] - results['mean_length'][0]:+.3f} characters
# - Initial mean length ({years[0]}): {results['mean_length'][0]:.3f} characters
# - Final mean length ({years[-1]}): {results['mean_length'][-1]:.3f} characters

# DISTRIBUTION CHARACTERISTICS:
# - Standard deviation range: {results['std_length'].min():.3f} - {results['std_length'].max():.3f}
# - Median length range: {results['median_length'].min():.1f} - {results['median_length'].max():.1f} characters

# TOP WORD LENGTHS BY VOCABULARY SIZE:
# """

#     # Add top word lengths by vocabulary size
#     vocab_by_length = corpus_stats['vocabulary_size_by_length']
#     top_lengths = sorted(vocab_by_length.items(), key=lambda x: x[1], reverse=True)[:10]

#     for length, count in top_lengths:
#         percentage = (count / corpus_stats['unique_words']) * 100
#         report += f"- {length} characters: {count:,} words ({percentage:.1f}% of vocabulary)\n"

#     return report


# def run_complete_analysis(corpus_df: pd.DataFrame) -> Dict[str, Any]:
#     """
#     Run the complete word length analysis pipeline.

#     Args:
#         corpus_df: DataFrame with 'AbstractText' and 'Year' columns

#     Returns:
#         Dictionary containing all analysis results
#     """
#     print("Step 1: Vectorizing corpus...")
#     X, words, years, counts, totals, df = vectorize_abstracts(corpus_df)

#     print("Step 2: Computing word length statistics...")
#     results = compute_word_length_statistics(df, years)

#     print("Step 3: Generating visualizations...")
#     fig = create_word_length_analysis_plots(results)

#     print("Step 4: Generating report...")
#     report = generate_word_length_report(results)

#     # Store additional metadata
#     results['vectorization_results'] = {
#         'count_matrix': X,
#         'words': words,
#         'years': years,
#         'counts': counts,
#         'totals': totals,
#         'df': df
#     }

#     results['report'] = report
#     results['figure'] = fig

#     return results

##########################################################################


if __name__ == "__main__":
    from pathlib import Path
    import time
    from datetime import timedelta

    # define paths
    variables_path = Path("../../results/variables")
    print(variables_path.resolve())
    # berenslab_data_path = Path("/gpfs01/berens/data/data/pubmed_processed")
    assert variables_path.exists(), "The path does not exist"
    # assert berenslab_data_path.exists(), "The path does not exist"

    # # load data
    # start = time.time()
    # df = load_data(start_year=2010, end_year=2025)
    # end = time.time()
    # runtime_total = end - start
    # print("Loading data runtime: ", str(timedelta(seconds=runtime_total)))

    # # clean up abstracts
    # start = time.time()
    # cleanup_abstracts_inplace(df)
    # end = time.time()
    # runtime_total = end - start
    # print("Cleaning abstracts runtime: ", str(timedelta(seconds=runtime_total)))

    # # compute yearly counts
    # start = time.time()
    years = np.arange(2010, 2026)
    # yearly_counts_df = vectorize_abstracts(df, years, variables_path)
    # end = time.time()
    # runtime_total = end - start
    # print("Yearly counts runtime: ", str(timedelta(seconds=runtime_total)))

    # compute word length statisctics
    ## load yearly counts df
    yearly_counts_df = pd.read_parquet(
        variables_path / "yearly-counts.parquet.gzip",
        engine="pyarrow",
    )
    start = time.time()
    results = compute_word_length_statistics(
        yearly_counts_df, saving_path=variables_path, years=years
    )
    end = time.time()
    runtime_total = end - start
    print("Word length statistics runtime: ", str(timedelta(seconds=runtime_total)))
