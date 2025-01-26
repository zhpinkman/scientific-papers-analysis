import collections as coll
import math
from multiprocessing import Pool, cpu_count
from collections import Counter, defaultdict
import pickle
import string
from binoculars import Binoculars
from spacy import displacy
import argparse
import json
from IPython import embed
from bs4 import BeautifulSoup
from collections import defaultdict
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import nltk
import spacy
from datetime import datetime, timezone
from profiling_decorator import profile


nlp = spacy.load("en_core_web_sm")

# Set up multiprocessing pool
import multiprocessing


def get_liwc_dictionary():
    LIWC_CATEGORIES_DICT = {}
    with open("liwc_dictionary.dic") as f:
        lines = f.readlines()
    num_of_percent_signs = 0
    liwc_dict = defaultdict(list)

    for index, line in enumerate(lines):
        if line.strip() == "%":
            num_of_percent_signs += 1
            continue
        elif num_of_percent_signs == 1:
            category_index, category_name = line.split("\t")
            LIWC_CATEGORIES_DICT[int(category_index)] = category_name.strip()
            continue

        word_plus_categories = line.split("\t")
        word = word_plus_categories[0]
        categories = word_plus_categories[1:]
        for category in categories:
            liwc_dict[LIWC_CATEGORIES_DICT[int(category)]].append(word.lower())
    return liwc_dict, LIWC_CATEGORIES_DICT


liwc_dict, LIWC_CATEGORIES_DICT = get_liwc_dictionary()
ALL_LIWC_WORDS_SET = set(
    [word.replace("*", "") for values in liwc_dict.values() for word in values]
)


def filter_text_based_on_liwc(text):
    text_words = set(text.split())
    filtered_text = text_words.intersection(ALL_LIWC_WORDS_SET)
    return filtered_text


def process_category(category, texts):
    print(f"Processing category {category}")
    category_counts = []
    for text in texts:
        try:
            text = f" {text} "
            text_length = len(text.split())
            count = 0
            for word in liwc_dict[category]:
                if word.endswith("*"):
                    count += text.count(f" {word[:-1]}")
                else:
                    count += text.count(f" {word} ")
            category_counts.append(count / text_length)
        except Exception as e:
            print(e)
            category_counts.append(np.nan)
    return category, category_counts


def add_liwc_features(input_df, input_text_column, LIWC_CATEGORIES_DICT):
    print("Adding LIWC features")
    output_df = input_df.copy()

    # Get all texts upfront
    texts = [
        " ".join(str(row[input_text_column]).lower().split())
        for _, row in output_df.iterrows()
    ]

    # Create pool and process categories in parallel
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # Map each category to be processed in parallel
    results = pool.starmap(
        process_category,
        [(cat, texts) for cat in LIWC_CATEGORIES_DICT.values()],
    )

    pool.close()
    pool.join()

    # Combine results into features dict
    liwc_features_dict = defaultdict(list)
    for category, counts in results:
        liwc_features_dict[category] = counts

    for category in LIWC_CATEGORIES_DICT.values():
        output_df[f"liwc_{category}"] = liwc_features_dict[category]

    return output_df


# Average Word Length
# Average Sentence Length By Word
# Average Sentence Length By Character
# Special Character Count
# Average Syllable per Word
# Functional Words Count
# Punctuation Count

## lexical, stylometric, and syntactic features


def syllable_count_Manual(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
            if word.endswith("e"):
                count -= 1
    if count == 0:
        count += 1
    return count


def syllable_count(word):
    global cmuDictionary
    d = cmuDictionary
    try:
        syl = [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    except:
        syl = syllable_count_Manual(word)
    return syl


# removing stop words plus punctuation.
def avg_wordLength(str):
    tokens = word_tokenize(str, language="english")
    st = [
        ",",
        ".",
        "'",
        "!",
        '"',
        "#",
        "$",
        "%",
        "&",
        "(",
        ")",
        "*",
        "+",
        "-",
        ".",
        "/",
        ":",
        ";",
        "<",
        "=",
        ">",
        "?",
        "@",
        "[",
        "\\",
        "]",
        "^",
        "_",
        "`",
        "{",
        "|",
        "}",
        "~",
        "\t",
        "\n",
    ]
    stop = stopwords.words("english") + st
    words = [word for word in tokens if word not in stop]
    return np.average([len(word) for word in words])


# ----------------------------------------------------------------------------


# returns avg number of characters in a sentence
def avg_SentLenghtByCh(text):
    tokens = sent_tokenize(text)
    return np.average([len(token) for token in tokens])


# ----------------------------------------------------------------------------


# returns avg number of words in a sentence
def avg_SentLenghtByWord(text):
    tokens = sent_tokenize(text)
    return np.average([len(token.split()) for token in tokens])


# ----------------------------------------------------------------------------
# GIVES NUMBER OF SYLLABLES PER WORD
def avg_Syllable_per_Word(text):
    tokens = word_tokenize(text, language="english")
    st = [
        ",",
        ".",
        "'",
        "!",
        '"',
        "#",
        "$",
        "%",
        "&",
        "(",
        ")",
        "*",
        "+",
        "-",
        ".",
        "/",
        ":",
        ";",
        "<",
        "=",
        ">",
        "?",
        "@",
        "[",
        "\\",
        "]",
        "^",
        "_",
        "`",
        "{",
        "|",
        "}",
        "~",
        "\t",
        "\n",
    ]
    stop = stopwords.words("english") + st
    words = [word for word in tokens if word not in stop]
    syllabls = [syllable_count(word) for word in words]
    p = " ".join(words)
    return sum(syllabls) / max(1, len(words))


# COUNTS SPECIAL CHARACTERS NORMALIZED OVER LENGTH OF CHUNK
def countSpecialCharacter(text):
    st = [
        "#",
        "$",
        "%",
        "&",
        "(",
        ")",
        "*",
        "+",
        "-",
        "/",
        "<",
        "=",
        ">",
        "@",
        "[",
        "\\",
        "]",
        "^",
        "_",
        "`",
        "{",
        "|",
        "}",
        "~",
        "\t",
        "\n",
    ]
    count = 0
    for character in st:
        count += text.count(character)

    return count / len(text.split())


# ----------------------------------------------------------------------------


def countPuncuation(text):
    st = [",", ".", "'", "!", '"', ";", "?", ":", ";"]
    count = 0
    for character in st:
        count += text.count(character)

    return float(count) / float(len(text.split()))


def RemoveSpecialCHs(text):
    text = word_tokenize(text)
    st = [
        ",",
        ".",
        "'",
        "!",
        '"',
        "#",
        "$",
        "%",
        "&",
        "(",
        ")",
        "*",
        "+",
        "-",
        ".",
        "/",
        ":",
        ";",
        "<",
        "=",
        ">",
        "?",
        "@",
        "[",
        "\\",
        "]",
        "^",
        "_",
        "`",
        "{",
        "|",
        "}",
        "~",
        "\t",
        "\n",
    ]

    words = [word for word in text if word not in st]
    return words


def CountFunctionalWords(text):
    functional_words = """a between in nor some upon
    about both including nothing somebody us
    above but inside of someone used
    after by into off something via
    all can is on such we
    although cos it once than what
    am do its one that whatever
    among down latter onto the when
    an each less opposite their where
    and either like or them whether
    another enough little our these which
    any every lots outside they while
    anybody everybody many over this who
    anyone everyone me own those whoever
    anything everything more past though whom
    are few most per through whose
    around following much plenty till will
    as for must plus to with
    at from my regarding toward within
    be have near same towards without
    because he need several under worth
    before her neither she unless would
    behind him no should unlike yes
    below i nobody since until you
    beside if none so up your
    """

    functional_words = functional_words.split()
    words = RemoveSpecialCHs(text)
    count = 0

    text = f" {text} "
    for word in functional_words:
        count += text.lower().count(f" {word} ")

    return count / len(words)


def count_functional_words_one_by_one(text):
    functional_words = """a between in nor some upon
    about both including nothing somebody us
    above but inside of someone used
    after by into off something via
    all can is on such we
    although cos it once than what
    am do its one that whatever
    among down latter onto the when
    an each less opposite their where
    and either like or them whether
    another enough little our these which
    any every lots outside they while
    anybody everybody many over this who
    anyone everyone me own those whoever
    anything everything more past though whom
    are few most per through whose
    around following much plenty till will
    as for must plus to with
    at from my regarding toward within
    be have near same towards without
    because he need several under worth
    before her neither she unless would
    behind him no should unlike yes
    below i nobody since until you
    beside if none so up your
    """
    text = f" {text} "
    functional_words = functional_words.split()
    functional_words_counts = dict()
    for word in functional_words:
        functional_words_counts[word] = text.lower().count(f" {word} ")

    normalized_counts = {
        word: count / len(text.split())
        for word, count in functional_words_counts.items()
    }
    return normalized_counts


# also returns Honore Measure R
def hapaxLegemena(text):
    words = RemoveSpecialCHs(text)
    V1 = 0
    # dictionary comprehension . har word kay against value 0 kardi
    freqs = {key: 0 for key in words}
    for word in words:
        freqs[word] += 1
    for word in freqs:
        if freqs[word] == 1:
            V1 += 1
    N = len(words)
    V = float(len(set(words)))
    R = 100 * math.log(N) / max(1, (1 - (V1 / V)))
    h = V1 / N
    return R, h


def hapaxDisLegemena(text):
    words = RemoveSpecialCHs(text)
    count = 0
    # Collections as coll Counter takes an iterable collapse duplicate and counts as
    # a dictionary how many equivelant items has been entered
    freqs = coll.Counter()
    freqs.update(words)
    for word in freqs:
        if freqs[word] == 2:
            count += 1

    h = count / float(len(words))
    S = count / float(len(set(words)))
    return S, h


# c(w)  = ceil (log2 (f(w*)/f(w))) f(w*) frequency of most commonly used words f(w) frequency of word w
# measure of vocabulary richness and connected to zipfs law, f(w*) const rak kay zips law say rank nikal rahay hein
def AvgWordFrequencyClass(text):
    words = RemoveSpecialCHs(text)
    # dictionary comprehension . har word kay against value 0 kardi
    freqs = {key: 0 for key in words}
    for word in words:
        freqs[word] += 1
    maximum = float(max(list(freqs.values())))
    return np.average(
        [math.floor(math.log((maximum + 1) / (freqs[word]) + 1, 2)) for word in words]
    )


# --------------------------------------------------------------------------
# TYPE TOKEN RATIO NO OF DIFFERENT WORDS / NO OF WORDS
def typeTokenRatio(text):
    words = word_tokenize(text)
    return len(set(words)) / len(words)


def get_lexical_density(text):
    num_functional_words = CountFunctionalWords(text)
    num_words = len(RemoveSpecialCHs(text))
    num_content_words = num_words - num_functional_words
    return num_content_words / num_words


### Vocabulary Richness
# --------------------------------------------------------------------------
# logW = V-a/log(N)
# N = total words , V = vocabulary richness (unique words) ,  a=0.17
# we can convert into log because we are only comparing different texts
def BrunetsMeasureW(text):
    words = RemoveSpecialCHs(text)
    a = 0.17
    V = float(len(set(words)))
    N = len(words)
    B = (V - a) / (math.log(N))
    return B


# -------------------------------------------------------------------------
# K  10,000 * (M - N) / N**2
# , where M  Sigma i**2 * Vi.
def YulesCharacteristicK(text):
    words = RemoveSpecialCHs(text)
    N = len(words)
    freqs = coll.Counter()
    freqs.update(words)
    vi = coll.Counter()
    vi.update(freqs.values())
    M = sum([(value * value) * vi[value] for key, value in freqs.items()])
    K = 10000 * (M - N) / math.pow(N, 2)
    return K


# -1*sigma(pi*lnpi)
# Shannon and sympsons index are basically diversity indices for any community
def ShannonEntropy(text):
    words = RemoveSpecialCHs(text)
    lenght = len(words)
    freqs = coll.Counter()
    freqs.update(words)
    arr = np.array(list(freqs.values()))
    distribution = 1.0 * arr
    distribution /= max(1, lenght)
    import scipy as sc

    H = sc.stats.entropy(distribution, base=2)
    # H = sum([(i/lenght)*math.log(i/lenght,math.e) for i in freqs.values()])
    return H


# ------------------------------------------------------------------
# 1 - (sigma(n(n - 1))/N(N-1)
# N is total number of words
# n is the number of each type of word
def SimpsonsIndex(text):
    words = RemoveSpecialCHs(text)
    freqs = coll.Counter()
    freqs.update(words)
    N = len(words)
    n = sum([1.0 * i * (i - 1) for i in freqs.values()])
    D = 1 - (n / (N * (N - 1)))
    return D


# def FleschReadingEase(text):
#     NoOfsentences = len(sent_tokenize(text))
#     words = RemoveSpecialCHs(text)
#     l = float(len(words))
#     scount = 0
#     for word in words:
#         scount += syllable_count(word)

#     I = 206.835 - 1.015 * (l / float(NoOfsentences)) - 84.6 * (scount / float(l))
#     return I


# -------------------------------------------------------------------
# def FleschCincadeGradeLevel(text):
#     NoOfSentences = len(sent_tokenize(text))
#     words = RemoveSpecialCHs(text)
#     scount = 0
#     for word in words:
#         scount += syllable_count(word)

#     l = len(words)
#     F = 0.39 * (l / NoOfSentences) + 11.8 * (scount / float(l)) - 15.59
#     return F


# -----------------------------------------------------------------
# def dale_chall_readability_formula(text):
#     NoOfSectences = len(sent_tokenize(text))
#     words = RemoveSpecialCHs(text)
#     difficult = 0
#     adjusted = 0
#     NoOfWords = len(words)
#     with open("data/dale-chall.pkl", "rb") as f:
#         fimiliarWords = pickle.load(f)
#     for word in words:
#         if word not in fimiliarWords:
#             difficult += 1
#     percent = (difficult / NoOfWords) * 100
#     if percent > 5:
#         adjusted = 3.6365
#     D = 0.1579 * (percent) + 0.0496 * (NoOfWords / NoOfSectences) + adjusted
#     return D


# ------------------------------------------------------------------
# def GunningFoxIndex(text):
#     NoOfSentences = len(sent_tokenize(text))
#     words = RemoveSpecialCHs(text)
#     NoOFWords = float(len(words))
#     complexWords = 0
#     for word in words:
#         if syllable_count(word) > 2:
#             complexWords += 1

#     G = 0.4 * ((NoOFWords / NoOfSentences) + 100 * (complexWords / NoOFWords))
#     return G


def compute_avg_dependency_link_length(text):
    doc = nlp(text)
    link_lengths = []
    for sent in doc.sents:
        sent_link_lengths = []
        for token in sent:
            if token.dep_ != "ROOT":
                head = token.head
                sent_link_lengths.append(abs(head.i - token.i))
        if sent_link_lengths:  # Only append if sentence had any links
            link_lengths.append(np.mean(sent_link_lengths))
    return np.mean(link_lengths)


def find_depth(token):
    if not list(token.children):  # If no children, the depth is 1
        return 1
    return 1 + max(find_depth(child) for child in token.children)


def find_deepest_trajectory(token):
    """
    Recursive helper function to find the trajectory of the deepest connection in a subtree.

    Parameters:
        token (spacy.tokens.Token): The root of the subtree.

    Returns:
        tuple: (depth, trajectory) where
            depth (int): Depth of the deepest subtree.
            trajectory (list): Tokens along the deepest path.
    """
    if not list(token.children):  # If no children, the depth is 1
        return 1, [token]
    # Get the deepest trajectory from the children
    child_depths = [find_deepest_trajectory(child) for child in token.children]
    max_depth, deepest_trajectory = max(child_depths, key=lambda x: x[0])
    return 1 + max_depth, [token] + deepest_trajectory


def compute_sentence_depth(text):
    doc = nlp(text)
    depths = []
    for sent in doc.sents:
        root = [token for token in sent if token.head == token][0]
        depths.append(find_depth(root))
    return np.mean(depths)


def get_dep_information(text):

    nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)

    dep_tags = [token.dep_ for token in doc]

    dep_distribution = {
        tag: count / len(text.split()) for tag, count in Counter(dep_tags).items()
    }

    return dep_distribution


def get_pos_information(text):

    # Load the SpaCy model
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    # Extract UD POS tags for each token
    pos_tags = [token.pos_ for token in doc]

    # Count the frequency of each POS tag
    pos_distribution = {
        tag: count / len(text.split()) for tag, count in Counter(pos_tags).items()
    }

    # Extract named entities
    ner_tags = [ent.label_ for ent in doc.ents]
    ner_distribution = {
        tag: count / len(text.split()) for tag, count in Counter(ner_tags).items()
    }

    return {"pos": pos_distribution, "ner": ner_distribution}


def test_get_pos_information():
    test_texts = [
        "I am thrilled to announce my promotion! However, I feel a bit anxious about the new challenges ahead.",
        "We met our friends at the park, and everyone had a great time talking and laughing.",
        "I think the solution is correct, but I’m unsure if the method aligns with the instructions.",
        "Yesterday was exhausting, but today feels manageable. I hope tomorrow brings more energy.",
        "My headache is unbearable, and I need some rest. Hopefully, drinking water will help.",
        # something with more proper nouns or company names or something like
        "I work at Google and I am happy to announce that we have launched a new product.",
        "James and Mary went to the park and had a great time.",
    ]
    for text in test_texts:
        print("Text:", text)
        print(get_pos_information(text))
        print("-" * 50)


def test_compute_sentence_depth():
    sample_texts = [
        "This is a sentence.",
        "I am thrilled to announce my promotion! However, I feel a bit anxious about the new challenges ahead.",
        "We met our friends at the park, and everyone had a great time talking and laughing.",
        "I think the solution is correct, but I’m unsure if the method aligns with the instructions.",
        "Yesterday was exhausting, but today feels manageable. I hope tomorrow brings more energy.",
        "My headache is unbearable, and I need some rest. Hopefully, drinking water will help.",
    ]
    doc = nlp(sample_texts[-1])
    embed()
    exit()
    # find the maximum number of hops from a root to a leaf
    max_depth = 0
    root = [token for token in doc if token.head == token][0]
    _, trajectory = find_deepest_trajectory(root)
    max_depth = find_depth(root)
    print("Max Depth:", max_depth)

    trajectory_tokens = [token.text for token in trajectory]
    print(f"Deepest trajectory: {' -> '.join(trajectory_tokens)}")

    displacy.serve(doc, style="dep", auto_select_port=True)


def test_process_syntactical_information():
    sample_texts = [
        "I am thrilled to announce my promotion! However, I feel a bit anxious about the new challenges ahead.",
        "We met our friends at the park, and everyone had a great time talking and laughing.",
        "I think the solution is correct, but I’m unsure if the method aligns with the instructions.",
        "Yesterday was exhausting, but today feels manageable. I hope tomorrow brings more energy.",
        "My headache is unbearable, and I need some rest. Hopefully, drinking water will help.",
    ]

    # get the dependency tree from the sample_texts[0] using spacy

    doc = nlp(sample_texts[0])
    for token in doc:
        print(
            token.text,
            token.dep_,
            token.head.text,
            token.head.pos_,
            [child for child in token.children],
        )

    embed()
    exit()


# returns a feature vector of text
def FeatureExtration(text):
    # cmu dictionary for syllables
    # global cmuDictionary
    # cmuDictionary = cmudict.dict()

    vector = {}
    try:
        # LEXICAL FEATURES
        vector["lex_avg_word_length"] = avg_wordLength(text)
        vector["lex_avg_sent_length_by_char"] = avg_SentLenghtByCh(text)
        vector["lex_avg_sent_length_by_word"] = avg_SentLenghtByWord(text)
        # vector["lex_avg_syllable_per_word"] = avg_Syllable_per_Word(text)
        vector["lex_special_char_count"] = countSpecialCharacter(text)
        vector["lex_punctuation_count"] = countPuncuation(text)
        vector["lex_functional_words_count"] = CountFunctionalWords(text)
        all_functional_words_counts = count_functional_words_one_by_one(text)
        for word, count in all_functional_words_counts.items():
            vector[f"lex_functional_word_{word}"] = count
        vector["lex_lexical_density"] = get_lexical_density(text)
        vector["lex_avg_dependency_link_length"] = compute_avg_dependency_link_length(
            text
        )
        vector["lex_sentence_depth"] = compute_sentence_depth(text)
        pos_ner_information = get_pos_information(text)
        for pos_tag, count in pos_ner_information["pos"].items():
            vector[f"lex_pos_{pos_tag}"] = count
        for ner_tag, count in pos_ner_information["ner"].items():
            vector[f"lex_ner_{ner_tag}"] = count

        dep_information = get_dep_information(text)
        for dep_tag, count in dep_information.items():
            vector[f"lex_dep_{dep_tag}"] = count

        # VOCABULARY RICHNESS FEATURES
        vector["voc_type_token_ratio"] = typeTokenRatio(text)

        HonoreMeasureR, hapax = hapaxLegemena(text)
        vector["voc_hapax_legomena"] = hapax
        vector["voc_honore_measure_r"] = HonoreMeasureR

        SichelesMeasureS, dihapax = hapaxDisLegemena(text)
        vector["voc_hapax_dislegomena"] = dihapax
        vector["voc_sichel_measure_s"] = SichelesMeasureS

        vector["voc_yule_k"] = YulesCharacteristicK(text)
        vector["voc_simpson_index"] = SimpsonsIndex(text)
        vector["voc_brunet_measure_w"] = BrunetsMeasureW(text)
        vector["voc_shannon_entropy"] = ShannonEntropy(text)
    except Exception as e:
        print(e)
        print("Error processing text:", text)

    # READIBILTY FEATURES
    # vector["read_flesch_reading_ease"] = FleschReadingEase(text)
    # vector["read_flesch_kincaid_grade"] = FleschCincadeGradeLevel(text)
    # vector["read_dale_chall_readability"] = dale_chall_readability_formula(text)
    # vector["read_gunning_fog_index"] = GunningFoxIndex(text)

    return vector


def compute_all_features_for_df(df, text_column):
    features = []
    # remove rows that are empty
    embed()
    exit()
    df = df.dropna(subset=[text_column])
    # convert the text_column to string
    df[text_column] = df[text_column].astype(str)
    df = df[df[text_column].apply(lambda x: len(x.split()) > 10)]
    # Create pool and process texts in parallel
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # Convert df to list of texts for parallel processing
    texts = [row[text_column] for _, row in df.iterrows()]

    # Process texts in parallel and collect features
    features = list(
        tqdm(
            pool.imap(FeatureExtration, texts),
            total=len(texts),
            desc="Extracting features",
        )
    )

    pool.close()
    pool.join()

    features_df = pd.DataFrame(features)
    for col in df:
        if col not in features_df:
            features_df[col] = df[col]
    # add liwc features
    features_df = add_liwc_features(features_df, text_column, LIWC_CATEGORIES_DICT)

    return features_df


def compute_similarities_based_on_features(features_df, year_col, month_col):

    grouped = features_df.groupby([year_col, month_col])
    features_columns = [
        col
        for col in features_df.columns
        if (col.startswith("voc_") or col.startswith("lex_") or col.startswith("liwc_"))
    ]

    months = []
    years = []
    variances = defaultdict(list)
    means = defaultdict(list)

    for name, group in grouped:
        year, month = name
        print(f"Computing similarities for year {year}, month {month}")
        for col in features_columns:
            variances[col].append(group[col].var())
            means[col].append(group[col].mean())
        months.append(month)
        years.append(year)

    variances_df = pd.DataFrame({"year": years, "month": months})
    for col in features_columns:
        variances_df[f"similarity_{col}"] = variances[col]
        variances_df[f"mean_{col}"] = means[col]

    return variances_df


@profile
def process_reddit_for_features(skip_to_similarity=False):
    if not skip_to_similarity:
        df = pd.read_csv("data/reddit/filtered_comments.csv")

        # grouped = df.groupby([df["year"], df["month"]])
        # text_groups = [(name, group["body"].tolist()) for name, group in grouped]

        features_df = compute_all_features_for_df(df, "body")
        features_df.to_csv("data/reddit/reddit_features.csv", index=False)

    else:
        features_df = pd.read_csv("data/reddit/reddit_features.csv")

    # Compute similarities based on features
    similarities_df = compute_similarities_based_on_features(
        features_df, "year", "month"
    )
    similarities_df.to_csv("data/reddit/reddit_similarities.csv", index=False)
    clean_similarities_df = clean_similarities(similarities_df)
    clean_similarities_df.to_csv(
        "data/reddit/reddit_clean_similarities.csv", index=False
    )


import language_tool_python


def process_chunk(texts):
    local_tool = language_tool_python.LanguageTool("en-US")
    chunk_errors = []
    for text in tqdm(texts, leave=False):
        try:
            num_errors = len(local_tool.check(text))
            chunk_errors.append(num_errors)
        except Exception as e:
            print(e)
            chunk_errors.append(np.nan)
    return chunk_errors


def check_for_grammatical_errors_reddit():
    df = pd.read_csv("data/reddit/filtered_comments.csv")

    all_texts = df["body"].tolist()

    # Split texts into chunks based on CPU count
    num_processes = cpu_count()
    chunk_size = len(all_texts) // num_processes
    chunks = [
        all_texts[i : min(i + chunk_size, len(all_texts))]
        for i in range(0, len(all_texts), chunk_size)
    ]

    # Process chunks in parallel
    with Pool(processes=num_processes) as pool:
        chunk_results = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks)))

    # Combine results
    all_num_errors = []
    for result in chunk_results:
        all_num_errors.extend(result)
    df["num_errors"] = all_num_errors
    df.to_csv("data/reddit/reddit_errors.csv", index=False)


def cache_news_data():
    with open("data/news/[tagged-zip]patchData.news.json") as f:
        data = json.load(f)
        f.close()

    # use beautiful soup to only get the content of the texts
    texts = [
        BeautifulSoup(d["body"], "html.parser").get_text()
        for d in tqdm(data, leave=False)
    ]

    update_times_days = [int(d["updated"][8:10]) for d in data]
    update_times_months = [int(d["updated"][5:7]) for d in data]
    update_times_years = [int(d["updated"][:4]) for d in data]

    df = pd.DataFrame(
        {
            "text": texts,
            "year": update_times_years,
            "month": update_times_months,
            "day": update_times_days,
        }
    )
    df.to_csv("data/news/news_data.csv", index=False)


def check_if_ai_written_news():
    with open("data/news/[tagged-zip]patchData.news.json") as f:
        data = json.load(f)
        f.close()

    # use beautiful soup to only get the content of the texts
    texts = [
        BeautifulSoup(d["body"], "html.parser").get_text()
        for d in tqdm(data, leave=False)
    ]

    update_times_days = [int(d["updated"][8:10]) for d in data]
    update_times_months = [int(d["updated"][5:7]) for d in data]
    update_times_years = [int(d["updated"][:4]) for d in data]

    df = pd.DataFrame(
        {
            "text": texts,
            "year": update_times_years,
            "month": update_times_months,
            "day": update_times_days,
        }
    )

    sampled_df = (
        df.groupby(["year", "month"])
        .apply(lambda x: x.sample(frac=0.15, replace=False, random_state=42))
        .reset_index(drop=True)
    )

    all_texts = sampled_df["text"].tolist()
    chunk_size = 32
    chunks = [
        all_texts[i : min(i + chunk_size, len(all_texts))]
        for i in range(0, len(all_texts), chunk_size)
    ]

    bino = Binoculars(DEVICE_1="cuda:0", DEVICE_2="cuda:1")

    all_results = []
    for chunk in tqdm(chunks, desc="Processing chunks"):
        try:
            chunk_results = bino.compute_score(chunk)
            all_results.extend(chunk_results)
        except Exception as e:
            print(e)
            all_results.extend([np.nan] * len(chunk))
            continue

    sampled_df["ai_written"] = all_results
    sampled_df.to_csv("data/news/news_ai_written.csv", index=False)


def check_if_ai_written_reddit():
    df = pd.read_csv("data/reddit/filtered_comments.csv")

    # for each month and year, sample 10% of the data
    sampled_df = (
        df.groupby(["year", "month"])
        .apply(lambda x: x.sample(frac=0.1, random_state=42))
        .reset_index(drop=True)
    )

    all_texts = sampled_df["body"].tolist()

    chunk_size = 32
    chunks = [
        all_texts[i : min(i + chunk_size, len(all_texts))]
        for i in range(0, len(all_texts), chunk_size)
    ]

    bino = Binoculars(DEVICE_1="cuda:0", DEVICE_2="cuda:1")

    all_results = []

    for chunk in tqdm(chunks, desc="Processing chunks"):
        try:
            chunk_results = bino.compute_score(chunk)
            all_results.extend(chunk_results)
        except Exception as e:
            print(e)
            all_results.extend([np.nan] * len(chunk))
            continue

    sampled_df["ai_written"] = all_results
    sampled_df.to_csv("data/reddit/reddit_ai_written.csv", index=False)


def check_if_ai_written_papers():

    bino = Binoculars(DEVICE_1="cuda:0", DEVICE_2="cuda:1")

    df = pd.read_csv("data/papers/cl_cv_papers.csv")
    df["final_date"] = pd.to_datetime(df["update_date"])
    df["year"] = df["final_date"].dt.year
    df["month"] = df["final_date"].dt.month

    all_texts = df["abstract"].tolist()

    # Split texts into 64 chunks
    chunk_size = 32
    chunks = [
        all_texts[i : min(i + chunk_size, len(all_texts))]
        for i in range(0, len(all_texts), chunk_size)
    ]

    # Process chunks and collect results
    all_results = []
    for chunk in tqdm(chunks, desc="Processing chunks"):
        try:
            chunk_results = bino.compute_score(chunk)
            all_results.extend(chunk_results)
        except Exception as e:
            print(e)
            # If chunk fails, fill with NaN for each text in chunk
            all_results.extend([np.nan] * len(chunk))
            continue

    df["ai_written"] = all_results
    df.to_csv("data/papers/cl_cv_papers_ai_written.csv", index=False)


def check_for_grammatical_errors_papers():
    df = pd.read_csv("data/papers/cl_cv_papers.csv")
    df["final_date"] = pd.to_datetime(df["update_date"])
    df["year"] = df["final_date"].dt.year
    df["month"] = df["final_date"].dt.month

    all_texts = df["abstract"].tolist()
    tool = language_tool_python.LanguageTool("en-US")
    all_num_errors = []
    for text in tqdm(all_texts):
        try:
            num_errors = len(tool.check(text))
            all_num_errors.append(num_errors)
        except Exception as e:
            print(e)
            all_num_errors.append(np.nan)
            continue
    df["num_errors"] = all_num_errors
    df.to_csv("data/papers/cl_cv_papers_errors.csv", index=False)


def check_for_grammatical_errors_news():
    with open("data/news/[tagged-zip]patchData.news.json") as f:
        data = json.load(f)
        f.close()

    # use beautiful soup to only get the content of the texts
    texts = [
        BeautifulSoup(d["body"], "html.parser").get_text()
        for d in tqdm(data, leave=False)
    ]

    update_times_months = [int(d["updated"][5:7]) for d in data]
    update_times_years = [int(d["updated"][:4]) for d in data]

    df = pd.DataFrame(
        {"text": texts, "year": update_times_years, "month": update_times_months}
    )

    df = (
        df.groupby(["year", "month"])
        .apply(lambda x: x.sample(frac=0.15, replace=False, random_state=42))
        .reset_index(drop=True)
    )

    all_texts = df["text"].tolist()

    num_processes = cpu_count()
    chunk_size = len(all_texts) // num_processes
    chunks = [
        all_texts[i : min(i + chunk_size, len(all_texts))]
        for i in range(0, len(all_texts), chunk_size)
    ]

    # Process chunks in parallel
    with Pool(processes=num_processes) as pool:
        chunk_results = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks)))

    # Combine results
    all_num_errors = []
    for result in chunk_results:
        all_num_errors.extend(result)
    df["num_errors"] = all_num_errors

    df.to_csv("data/news/news_errors.csv", index=False)


def clean_similarities(data):
    data["year"] = data["year"].astype(float).astype(int)
    data["month"] = data["month"].astype(float).astype(int)

    # only keep the rows with year >= 2018
    data = data[data["year"] >= 2018]

    # sort the data by year and month
    # Fill NaN values with mean of each column
    data = data.fillna(data.mean())
    data = data.set_index(["year", "month"]).sort_index().reset_index()

    return data


@profile
def process_papers_for_features(skip_to_similarity=False):
    if not skip_to_similarity:
        df = pd.read_csv("data/papers/cl_cv_papers.csv")
        df["final_date"] = pd.to_datetime(df["update_date"])
        df["year"] = df["final_date"].dt.year
        df["month"] = df["final_date"].dt.month
        features_df = compute_all_features_for_df(df, "abstract")
        features_df.to_csv("data/papers/cl_cv_papers_features.csv", index=False)
    else:
        features_df = pd.read_csv("data/papers/cl_cv_papers_features.csv")

    # Compute similarities based on features
    similarities_df = compute_similarities_based_on_features(
        features_df, "year", "month"
    )
    similarities_df.to_csv("data/papers/cl_cv_papers_similarities.csv", index=False)
    clean_similarities_df = clean_similarities(similarities_df)
    clean_similarities_df.to_csv(
        "data/papers/cl_cv_papers_clean_similarities.csv", index=False
    )


@profile
def process_news_for_features(skip_to_similarity=False):
    if not skip_to_similarity:
        with open("data/news/[tagged-zip]patchData.news.json") as f:
            data = json.load(f)
            f.close()

        # use beautiful soup to only get the content of the texts
        texts = [
            BeautifulSoup(d["body"], "html.parser").get_text()
            for d in tqdm(data, leave=False)
        ]

        update_times_days = [int(d["updated"][8:10]) for d in data]
        update_times_months = [int(d["updated"][5:7]) for d in data]
        update_times_years = [int(d["updated"][:4]) for d in data]

        df = pd.DataFrame(
            {
                "text": texts,
                "year": update_times_years,
                "month": update_times_months,
                "day": update_times_days,
            }
        )
        # from each (year, month), take 1 / 4 of the data
        df = (
            df.groupby(["year", "month"])
            .apply(lambda x: x.sample(frac=0.25, replace=False, random_state=42))
            .reset_index(drop=True)
        )

        features_df = compute_all_features_for_df(df, "text")
        features_df.to_csv("data/news/news_features.csv", index=False)
    else:
        features_df = pd.read_csv("data/news/news_features.csv")

    # Compute similarities based on features
    similarities_df = compute_similarities_based_on_features(
        features_df, "year", "month"
    )
    similarities_df.to_csv("data/news/news_similarities.csv", index=False)
    clean_similarities_df = clean_similarities(similarities_df)
    clean_similarities_df.to_csv("data/news/news_clean_similarities.csv", index=False)


def test_process_papers_for_features():

    liwc_test_texts = {
        "Emotion and Affect": "I am thrilled to announce my promotion! However, I feel a bit anxious about the new challenges ahead.",
        "Social Processes": "We met our friends at the park, and everyone had a great time talking and laughing.",
        "Cognitive Processes": "I think the solution is correct, but I’m unsure if the method aligns with the instructions.",
        "Temporal Focus": "Yesterday was exhausting, but today feels manageable. I hope tomorrow brings more energy.",
        "Biological and Health Concerns": "My headache is unbearable, and I need some rest. Hopefully, drinking water will help.",
    }

    df = pd.DataFrame(
        {
            "abstract": list(liwc_test_texts.values()),
            "label": list(liwc_test_texts.keys()),
        }
    )
    features_df = compute_all_features_for_df(df, "abstract")
    embed()
    exit()


def one_gram_similarity_over_liwc_words(text_1, text_2):
    text_1 = text_1.split()
    text_2 = text_2.split()
    text_1_liwc = set(text_1).intersection(set(liwc_dict.keys()))
    text_2_liwc = set(text_2).intersection(set(liwc_dict.keys()))


def n_gram_similarity(text_1, text_2, n=1):
    text_1 = text_1.split()
    text_2 = text_2.split()
    n_grams_text_1 = set(zip(*[text_1[i:] for i in range(n)]))
    n_grams_text_2 = set(zip(*[text_2[i:] for i in range(n)]))

    return len(n_grams_text_1.intersection(n_grams_text_2)) / len(
        n_grams_text_1.union(n_grams_text_2)
    )


def compute_similarities_in_group(group, texts):
    print(f"Computing similarities for group {group}")
    import string

    clean_texts = []

    for text in texts:
        clean_text = text
        # remove all punctuation and put space in place of it
        clean_text = clean_text.translate(
            str.maketrans(string.punctuation, " " * len(string.punctuation))
        )
        clean_text = clean_text.replace("\n", " ")
        # remove extra spaces
        clean_text = " ".join(clean_text.split())
        clean_text = clean_text.lower()
        clean_texts.append(clean_text)

    # using the n_gram_similarity function, parallelize the computation and compute the similarity between all pairs of texts
    similarities_dict = {}

    def process_n_gram(n, clean_texts):
        n_similarities = {}
        for i in tqdm(range(len(clean_texts)), leave=False):
            for j in range(i, len(clean_texts)):
                try:
                    n_similarities[(i, j)] = n_gram_similarity(
                        clean_texts[i], clean_texts[j], n
                    )
                except Exception as e:
                    n_similarities[(i, j)] = np.nan
        return n, n_similarities

    # Use threading instead of multiprocessing for nested parallelization
    import threading
    from concurrent.futures import ThreadPoolExecutor

    # Process each n-gram in parallel using threads
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks and get futures
        futures = [executor.submit(process_n_gram, n, clean_texts) for n in [1, 2, 3]]

        # Get results as they complete
        results = []
        for future in futures:
            results.append(future.result())

    # Combine results into similarities dict
    for n, n_similarities in results:
        similarities_dict[n] = n_similarities
    final_object = {
        "texts": texts,
        "clean_texts": clean_texts,
        "similarities": similarities_dict,
    }
    return final_object


def compute_n_gram_similarities_per_month(text_groups):

    processed_text_groups = []
    for name, group in text_groups:
        if len(group) < 1000:
            processed_text_groups.append((name, group))
        else:
            np.random.seed(42)
            processed_text_groups.append(
                (
                    name,
                    np.random.choice(
                        group,
                        1000,
                        replace=False,
                    ).tolist(),
                )
            )

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # Process groups in parallel and store results
    results = {}
    for name, similarities in zip(
        [g[0] for g in processed_text_groups],
        pool.starmap(compute_similarities_in_group, processed_text_groups),
    ):
        year, month = name
        print(
            f"Finished processing year {year}, month {month} with number of entities {len(similarities['texts'])}"
        )
        results[name] = similarities

    pool.close()
    pool.join()

    print("Completed processing all month-year groups")

    return results


def create_ngram_similarities_df(results):
    months = []
    years = []
    text_indices_0 = []
    text_indices_1 = []
    n_gram_type = []
    similarities = []

    for (year, month), similarities_dict in results.items():
        for n, n_similarities in similarities_dict["similarities"].items():
            for (i, j), similarity in n_similarities.items():
                months.append(month)
                years.append(year)
                text_indices_0.append(i)
                text_indices_1.append(j)
                n_gram_type.append(n)
                similarities.append(similarity)

    df = pd.DataFrame(
        {
            "year": years,
            "month": months,
            "text_index_0": text_indices_0,
            "text_index_1": text_indices_1,
            "n_gram_type": n_gram_type,
            "similarity": similarities,
        }
    )
    return df


def process_reddit_n_gram_similarities_per_month():
    df = pd.read_csv("data/reddit/filtered_comments.csv")

    grouped = df.groupby([df["year"], df["month"]])
    text_groups = [(name, group["body"].tolist()) for name, group in grouped]

    print(f"Found {len(text_groups)} month-year groups to process")
    print("Min entities in a group:", min([len(g[1]) for g in text_groups]))
    print("Max entities in a group:", max([len(g[1]) for g in text_groups]))
    print("Mean entities in a group:", np.mean([len(g[1]) for g in text_groups]))
    print("Median entities in a group:", np.median([len(g[1]) for g in text_groups]))

    results = compute_n_gram_similarities_per_month(text_groups)

    np.save("data/reddit/reddit_n_gram_similarities.npy", results)
    create_ngram_similarities_df(results).to_csv(
        "data/reddit/reddit_n_gram_similarities.csv", index=False
    )


def process_news_n_gram_similarities_per_month():
    with open("data/news/[tagged-zip]patchData.news.json") as f:
        data = json.load(f)
        f.close()

    # use beautiful soup to only get the content of the texts
    texts = [
        BeautifulSoup(d["body"], "html.parser").get_text()
        for d in tqdm(data, leave=False)
    ]

    # Group by month and year
    # based on "updated": "2023-10-20T14:43:37Z"
    update_times_months = [int(d["updated"][5:7]) for d in data]
    update_times_years = [int(d["updated"][:4]) for d in data]
    text_groups_df = pd.DataFrame(
        {
            "text": texts,
            "update_time_year": update_times_years,
            "update_time_month": update_times_months,
        }
    )
    grouped = text_groups_df.groupby(["update_time_year", "update_time_month"])

    # Create list of text groups to process
    text_groups = [(name, group["text"].tolist()) for name, group in grouped]
    print(f"Found {len(text_groups)} month-year groups to process")

    print("Min entities in a group:", min([len(g[1]) for g in text_groups]))
    print("Max entities in a group:", max([len(g[1]) for g in text_groups]))
    print("Mean entities in a group:", np.mean([len(g[1]) for g in text_groups]))
    print("Median entities in a group:", np.median([len(g[1]) for g in text_groups]))

    results = compute_n_gram_similarities_per_month(text_groups)

    np.save("data/news/news_n_gram_similarities.npy", results)
    create_ngram_similarities_df(results).to_csv(
        "data/news/news_n_gram_similarities.csv", index=False
    )


def process_papers_n_gram_similarities_per_month():
    df = pd.read_csv("data/papers/cl_cv_papers.csv")
    df["final_date"] = pd.to_datetime(df["update_date"])

    print("Starting n-gram similarity processing...")

    # Group by month and year
    grouped = df.groupby([df["final_date"].dt.year, df["final_date"].dt.month])

    # Create list of text groups to process
    text_groups = [(name, group["abstract"].tolist()) for name, group in grouped]
    print(f"Found {len(text_groups)} month-year groups to process")
    # show the mean, min and max of the entities in each group
    print("Min entities in a group:", min([len(g[1]) for g in text_groups]))
    print("Max entities in a group:", max([len(g[1]) for g in text_groups]))
    print("Mean entities in a group:", np.mean([len(g[1]) for g in text_groups]))
    print("Median entities in a group:", np.median([len(g[1]) for g in text_groups]))
    # for testing, limit the text_groups to only a few in each group
    # text_groups = [(name, group[:10]) for name, group in text_groups]

    results = compute_n_gram_similarities_per_month(text_groups)

    np.save("data/papers/papers_n_gram_similarities.npy", results)
    create_ngram_similarities_df(results).to_csv(
        "data/papers/papers_n_gram_similarities.csv", index=False
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--process", type=str, required=True, help="process papers for features"
    )
    parser.add_argument(
        "--skip_to_similarity",
        action="store_true",
        help="skip to similarity and don't recompute features",
    )

    args = parser.parse_args()

    if args.process == "papers_featurization":
        process_papers_for_features(args.skip_to_similarity)
    elif args.process == "papers_ngram_sim":
        process_papers_n_gram_similarities_per_month()
    elif args.process == "news_featurization":
        process_news_for_features(args.skip_to_similarity)
    elif args.process == "news_ngram_sim":
        process_news_n_gram_similarities_per_month()
    elif args.process == "reddit_ngram_sim":
        process_reddit_n_gram_similarities_per_month()
    elif args.process == "reddit_featurization":
        process_reddit_for_features(args.skip_to_similarity)
    elif args.process == "test_papers_featurization":
        test_process_papers_for_features()
    elif args.process == "test_syntactical_information":
        test_process_syntactical_information()
    elif args.process == "test_compute_sentence_depth":
        test_compute_sentence_depth()
    elif args.process == "test_get_pos_information":
        test_get_pos_information()
    elif args.process == "check_errors_papers":
        check_for_grammatical_errors_papers()
    elif args.process == "check_errors_reddit":
        check_for_grammatical_errors_reddit()
    elif args.process == "check_errors_news":
        check_for_grammatical_errors_news()
    elif args.process == "check_ai_written_papers":
        check_if_ai_written_papers()
    elif args.process == "check_ai_written_reddit":
        check_if_ai_written_reddit()
    elif args.process == "check_ai_written_news":
        check_if_ai_written_news()
    else:
        print("Invalid process argument")
        exit()
