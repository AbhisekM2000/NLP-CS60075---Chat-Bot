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


from helper import *
from process_data import *
from Vocabulary import vocab
from Vocabulary import vocab_helper


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS = ["character1ID",
                              "character2ID", "movieID", "utteranceIDs"]
MAX_LENGTH = 10  # Maximum sentence length to consider
MIN_COUNT = 3    # Minimum word count threshold for trimming
# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)

# Define path to new file
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

# Initialize lines dict, conversations list, and field ids
lines = {}
conversations = []

# Load lines and process conversations
print("\nProcessing corpus...")
lines = processLine(os.path.join(
    corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
print("\nLoading conversations...")
conversations = processConversation(os.path.join(corpus, "movie_conversations.txt"),
                                    lines, MOVIE_CONVERSATIONS_FIELDS)

# Write new csv file
print("\nWriting newly formatted file...")
with open(datafile, 'w', encoding='utf-8') as f:
    w = csv.writer(f, delimiter=str(codecs.decode(
        '\t', "unicode_escape")), lineterminator='\n')
    for qa in get_QA_pairs(conversations):
        w.writerow(qa)

# Print a sample of lines
print("\nSample lines from file:")
printLines(datafile)


# Load/Assemble vocabulary and pairs
save_dir = os.path.join("data", "save")
voc, pairs = vocab_helper.loadPrepareData(
    corpus, corpus_name, datafile, save_dir)
# Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)


# Trim vocabulary and pairs
pairs = vocab_helper.trimVocabulary(voc, pairs, MIN_COUNT)


# Example for validation
small_batch_size = 5
batches = vocab_helper.batch2TrainData(
    voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)
