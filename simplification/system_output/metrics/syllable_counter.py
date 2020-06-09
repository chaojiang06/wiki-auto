# -*- coding: utf-8 -*-

import re
from nltk.corpus import cmudict

"""
Based on code by Danielle Sucher
#https://github.com/DanielleSucher/Nantucket/blob/master/poetry.py
"""


class SyllableCounter:
    def __init__(self):
        self.syl_dict = cmudict.dict()
        self.more_syls = ["n't", "'re", "'s", "'ll", "'ve", "'d", "'m"]

    def get_feature(self, word):
        word = word.lower()
        if word in self.more_syls:
            return 1
        if word not in self.syl_dict:
            return self._approx_nsyl(word)
        return max([len(list(y for y in x if y[-1].isdigit())) for x in self.syl_dict[word]])

    @staticmethod
    def _approx_nsyl(word):
        digraphs = ["ai", "au", "ay", "ea", "ee", "ei", "ey", "oa", "oe", "oi", "oo", "ou", "oy", "ua", "ue", "ui"]
        # Ambiguous, currently split: ie, io
        # Ambiguous, currently kept together: ui
        digraphs = set(digraphs)
        count = 0
        array = re.split("[^aeiouy]+|[0-9]", word)

        for i, v in enumerate(array):
            if len(v) > 1 and v not in digraphs:
                count += 1
            if v == '':
                del array[i]
        count += len(array)
        if re.search("(?<=\w)(ion|ious|(?<=!t)ed|es|[^lr]e)(?![a-z']+)", word):
            count -= 1
        if re.search("'ve|n't", word):
            count += 1

        alldigits = True
        for ch in word:
            if ch.isdigit():
                count += 1
            else:
                alldigits = False

        if len(array) == 1:
            count = len(word)

        if alldigits:
            count = 1

        return count
