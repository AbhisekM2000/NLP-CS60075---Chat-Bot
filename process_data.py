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


# Splits each line of the file into a dictionary of fields  (lineID, characterID, movieID, character, text)
def processLine(fName, keys):
    #We define an empty dictionary
    lines = {}
    # Now we iterate through the fine with a proper encoding type
    with open(fName, 'r', encoding='iso-8859-1') as f:
        # We go through each line of the file
        for line in f:
            # We split the line by the tabulator
            values = line.split(" +++$+++ ")
            # We define an empty dictionary to store the contents of the field
            temp = {}
            # Then we iterate through the fileds and then add the contents to the dictionary
            for i, field in enumerate(keys):
                temp[field] = values[i]
            #Finally for that field we add the lineID as key and the dictionary as value, and that becomes the new key for lines, and the value is the dictionary
            lines[temp['lineID']] = temp
    return lines


# Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*, where fName is the path to the file, l is the line number and keys are the fields we want
def processConversation(fName, l, keys):
    # We define an empty list which we would return
    final = []
    # Now we iterate through the fine with a proper encoding type
    with open(fName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            # We split the line by the tabulator
            values = line.split(" +++$+++ ")
            # We define an empty dictionary to store the contents of the field
            conversation_obj = {}
            for i, field in enumerate(keys):
                conversation_obj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            conversation_obj["lines"] = []
            for id in re.compile('L[0-9]+').findall(conversation_obj["utteranceIDs"]):
                conversation_obj["lines"].append(l[id])
            #Finally we append the conversation object to the list
            final.append(conversation_obj)
    return final


# Extracts pairs of sentences from conversations, where conversations the list returned from the function above
def get_QA_pairs(conversations):
    final_pairs = []
    for CONV in conversations:
        # Iterate over all the lines of the conversation and we ignore the last line (no answer for it)
        for i in range(len(CONV["lines"]) - 1):
            # Filter wrong samples (if one of the lists is empty), thus we check if none are empty where the ith text is the input and the (i + 1)th text is the target
            if CONV["lines"][i]["text"].strip() and CONV["lines"][i+1]["text"].strip():
                final_pairs.append([CONV["lines"][i]["text"].strip(), CONV["lines"][i+1]["text"].strip()])
    # Then finally we return the QA pairs that we got
    return final_pairs
