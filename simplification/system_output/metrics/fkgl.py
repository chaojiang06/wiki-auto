import argparse
from string import punctuation
from syllable_counter import SyllableCounter
from nltk import sent_tokenize, word_tokenize

PUNCTUATION = set(punctuation)


def is_puntuation(word):
    return all([ch in PUNCTUATION for ch in word])


def get_counts(counter, text):
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    syl_num = sum([counter.get_feature(w) for w in words])
    return syl_num, len(words), len(sentences)


def get_fkgl_from_counts(word_count, sentence_count, syllable_count):
    avg_words_per_sent = (1.0 * word_count) / sentence_count
    avg_syll_per_word = (1.0 * syllable_count) / word_count
    return avg_words_per_sent, avg_syll_per_word, 0.39 * avg_words_per_sent + 11.8 * avg_syll_per_word - 15.59


def compute_fkgl(file_name):

    syllable_counter = SyllableCounter()
    text = [line.strip() for line in open(file_name)]

    words_count = 0
    syllables_count = 0
    sentences_count = 0

    for line in text:
        syl_count, w_count, sent_count = get_counts(syllable_counter, line)
        syllables_count += syl_count
        sentences_count += sent_count
        words_count += w_count

    avg_sent, avg_word, fkgl = get_fkgl_from_counts(words_count, sentences_count, syllables_count)
    return avg_sent, avg_word, fkgl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', "--simplified")
    args = parser.parse_args()
    print(compute_fkgl(args.input))
