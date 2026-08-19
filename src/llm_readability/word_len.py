# modified from repo llm-excess-vocab/scripts/02_preprocess_and_count_utils.py

from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt
from tqdm import tqdm


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


def vectorize_abstracts(df, years, saving_path, token_pattern=r"\b[a-zA-Z]+\b"):
    vectorizer = CountVectorizer(
        lowercase=True,
        token_pattern=token_pattern,  # Only alphabetic words
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


def compute_word_length_statistics(
    yearly_counts_df, saving_path, years=None, length_threshold=0
):
    """
    Efficiently compute word length statistics from word counts dataframe.

    Args:
        df: DataFrame with 'word' column and year columns containing counts
        years: Array of years corresponding to the year columns
        length_threshold: int, default=0
            Only words with length greater than that value are used for the analysis.

    Returns:
        Dictionary containing word length statistics and analysis results
    """
    if years is None:
        years = np.array(yearly_counts_df.columns[1:], dtype=int)

    # Remove the totals row (last row with empty word)
    word_counts_df = yearly_counts_df[yearly_counts_df["word"] != ""].copy()

    # Compute word lengths vectorized
    word_lengths = word_counts_df["word"].str.len().values

    # length threshold
    if length_threshold:
        mask = word_lengths > length_threshold
        print(f"{np.sum(mask)} out of {word_counts_df.shape[0]} words kept")
        print(f"{word_counts_df.shape[0]-np.sum(mask)} words excluded")
        word_counts_df = word_counts_df[mask]
        word_lengths = word_lengths[mask]

    # Extract count matrix (all year columns)
    year_columns = [str(year) for year in years]  # they are not str!!
    count_matrix = word_counts_df[year_columns].values

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
    for i, year in enumerate(tqdm(years)):
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

        # Count words by length category (vectorized)
        length_counts = np.bincount(
            word_lengths, weights=year_counts
        )  # NOTE: np.bincount will give you a count for each int, including 0, even if there is no word with len=0

        results["length_category_counts"][str(year)] = {
            "lengths": np.arange(word_lengths.max() + 1),
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
