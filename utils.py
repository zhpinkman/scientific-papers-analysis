import collections as coll
import math
import pickle
import string
from collections import defaultdict
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from nltk.corpus import cmudict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import nltk

nltk.download("cmudict")
nltk.download("stopwords")

style.use("ggplot")
cmuDictionary = None

## similarity metrics


def n_gram_similarity(text_1, text_2, n=1):
    text_1 = text_1.lower()
    text_2 = text_2.lower()
    text_1 = text_1.split()
    text_2 = text_2.split()
    n_grams_text_1 = set(zip(*[text_1[i:] for i in range(n)]))
    n_grams_text_2 = set(zip(*[text_2[i:] for i in range(n)]))

    return len(n_grams_text_1.intersection(n_grams_text_2)) / len(
        n_grams_text_1.union(n_grams_text_2)
    )


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


def add_liwc_features(input_df, input_text_column, liwc_dict, LIWC_CATEGORIES_DICT):
    print("Adding LIWC features")
    output_df = input_df.copy()

    liwc_features_dict = defaultdict(list)
    for _, row in tqdm(output_df.iterrows(), leave=False, total=len(output_df)):
        try:
            text = row[input_text_column]
            text = f" {text} "
            text_length = len(text.split())
            for category in LIWC_CATEGORIES_DICT.values():
                count = 0
                for word in liwc_dict[category]:
                    if word.endswith("*"):
                        count += text.count(f" {word[:-1]}")
                    else:
                        count += text.count(f" {word} ")
                liwc_features_dict[category].append(count / text_length)
        except Exception as e:
            print(e)
            for category in LIWC_CATEGORIES_DICT.values():
                liwc_features_dict[category].append(np.nan)

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
    str.translate(string.punctuation)
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
    for i in text:
        if i in st:
            count = count + 1
    return count / len(text)


# ----------------------------------------------------------------------------


def countPuncuation(text):
    st = [",", ".", "'", "!", '"', ";", "?", ":", ";"]
    count = 0
    for i in text:
        if i in st:
            count = count + 1
    return float(count) / float(len(text))


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

    for i in text:
        if i in functional_words:
            count += 1

    return count / len(words)


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


# returns a feature vector of text
def FeatureExtration(text):
    # cmu dictionary for syllables
    # global cmuDictionary
    # cmuDictionary = cmudict.dict()

    vector = {}

    # LEXICAL FEATURES
    vector["lex_avg_word_length"] = avg_wordLength(text)
    vector["lex_avg_sent_length_by_char"] = avg_SentLenghtByCh(text)
    vector["lex_avg_sent_length_by_word"] = avg_SentLenghtByWord(text)
    # vector["lex_avg_syllable_per_word"] = avg_Syllable_per_Word(text)
    vector["lex_special_char_count"] = countSpecialCharacter(text)
    vector["lex_punctuation_count"] = countPuncuation(text)
    vector["lex_functional_words_count"] = CountFunctionalWords(text)

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

    # READIBILTY FEATURES
    # vector["read_flesch_reading_ease"] = FleschReadingEase(text)
    # vector["read_flesch_kincaid_grade"] = FleschCincadeGradeLevel(text)
    # vector["read_dale_chall_readability"] = dale_chall_readability_formula(text)
    # vector["read_gunning_fog_index"] = GunningFoxIndex(text)

    return vector


def compute_all_features_for_df(df, text_column):
    features = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row[text_column].lower()
        features.append(FeatureExtration(text))

    features_df = pd.DataFrame(features)
    features_df["abstract"] = df["abstract"]
    # add liwc features
    liwc_dict, LIWC_CATEGORIES_DICT = get_liwc_dictionary()
    features_df = add_liwc_features(
        features_df, text_column, liwc_dict, LIWC_CATEGORIES_DICT
    )

    return features_df


from profiling_decorator import profile


@profile
def process_papers():
    df = pd.read_csv("data/papers/cl_cv_papers.csv")
    # small subset
    # df = df.sample(100)
    features_df = compute_all_features_for_df(df, "abstract")
    features_df.to_csv("data/papers/cl_cv_papers_features.csv", index=False)


if __name__ == "__main__":

    process_papers()
    # # Example texts with varying richness and readability
    # examples = {
    #     "Simple Text": "The cat sat on the mat. The dog barked. It was a sunny day.",
    #     "Moderate Richness": (
    #         "A feline reclined lazily upon the rug, basking in the warmth of the sun. "
    #         "Nearby, a canine emitted a sharp bark, breaking the tranquil atmosphere."
    #     ),
    #     "High Richness": (
    #         "An exquisite tabby luxuriated atop the intricately woven Persian carpet, "
    #         "soaking up golden sunbeams filtering through the gossamer curtains. "
    #         "In stark contrast, a spirited terrier issued an abrupt and piercing bark, "
    #         "shattering the serene ambiance."
    #     ),
    #     "Technical Text": (
    #         "The aqueous solution was heated to 100°C, initiating a phase transition "
    #         "from liquid to vapor, as described by the Clausius-Clapeyron relation."
    #     ),
    #     "Child-Friendly Text": (
    #         "The little bunny hopped and hopped. He saw a butterfly and smiled. The world was happy and fun!"
    #     ),
    # }
    # # Extract features for each example text
    # features = {name: FeatureExtration(text) for name, text in examples.items()}
    # features_df = pd.DataFrame(features).T
    # from IPython import embed

    # embed()
    # exit()
