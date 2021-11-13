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

# Its prints the contents of the file, the first 20 lines
def printLines(file, n=20):
    with open(file, 'rb') as f:
        lines = f.readlines()
    for line in lines[:n]:
        print(line)

# We normalize the strings , Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    # WE need to convert unicode to ASCII
    s=s.lower().strip()
    s=''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    # We use the regex function to remove unwanted characters
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    # Finally we return the normalized string
    return s