from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS = ["character1ID","character2ID", "movieID", "utteranceIDs"]
MAX_LENGTH = 10  # Maximum sentence length to consider
# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class MY_VOCABULARY:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.w2i = {} # Word to index
        self.w2c = {} # Word to count
        self.i2w = {PAD_token: "PAD",
                           SOS_token: "SOS", EOS_token: "EOS"} # index to word
        self.num_words = 3  # Count SOS, EOS, PAD tokens

    def add_sentence_to_vocabulary(self, sentence): # Function to add a sentence to the vocabulary
        for word in sentence.split(' '): # We iterate through the words of the sentence
            self.add_word_to_vocabulary(word)

    def add_word_to_vocabulary(self, word): # Function to add words to the vocabulary
        if word not in self.w2i: # If the word is not in the vocabulary then we add it
            self.w2i[word] = self.num_words
            self.w2c[word] = 1
            self.i2w[self.num_words] = word
            self.num_words += 1
        else:  # Else we add one to the word count since its already in the dictionary
            self.w2c[word] += 1

    # Remove words below a certain count threshold, that is we only keep the most common words
    def trim(self, min_count,to_print=True):
        if self.trimmed: # If already trimmed then we return
            return
        self.trimmed = True # Else we mark the bool as True
        most_frequent_words = [] # These are the words we cant to keep
        for k, v in self.w2c.items():
            if v >= min_count:
                most_frequent_words.append(k)
        if to_print==True:
            print('keep_words {} / {} = {:.4f}'.format(
                len(most_frequent_words), len(self.w2i), len(
                    most_frequent_words) / len(self.w2i)
            ))
        # Reinitialize the old dictionaries dictionaries
        self.w2i = {}
        self.w2c = {}
        self.i2w = {PAD_token: "PAD",
                           SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count default tokens

        for word in most_frequent_words:  # We iterate through the words of the sentence and add it to the vocabulary
            self.add_word_to_vocabulary(word)
