# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 10:04:10 2021

@author: NguyenSon
"""
from __future__ import print_function

import collections
from collections import Counter
import json
import os
import numpy as np
from re import compile as _Re
import string
import re

import sys
import neologdn


stopwords = []
path = r'D:\VBDI_NLP\bert\japanese_stopword_list.txt' # change path to read stopwords
with open(path, encoding = 'utf-8') as f1:
    for line in f1.read().splitlines():
        stopwords.append(line)
    f1.close()

def remove_stopword(text):
    splt = split_unicode_chrs(text)
    for ch in splt[:]:
        if ch in stopwords:
           splt.remove(ch)
    text_ = ''.join(splt)
    return text_

def normalize_answer_jp(s):
    text = neologdn.normalize(remove_stopword(s))
    return text

_unicode_chr_splitter = _Re( '(?s)((?:[\ud800-\udbff][\udc00-\udfff])|.)' ).split

def split_unicode_chrs( text ):
    '''
    Split unicode that related to Japanese or Chinese character 

    Parameters
    ----------
    text : TYPE string
        DESCRIPTION. 

    Returns
    -------
    list
        DESCRIPTION. List of splited characters from raw string

    '''
    return [ chr for chr in _unicode_chr_splitter( text ) if chr ]



def normalize_answer(s):
    '''
    Lower text and remove punctuation, articles and extra whitespace.
    Just use for English text 
    
    Parameters
    ----------
    s : TYPE string
        DESCRIPTION.

    Returns
    -------
    TYPE string
        DESCRIPTION. Normalized string 

    '''
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    '''
    F1-Score calculator for String sequence
    CAUTION: (only used for Japanese string, it may be used for Chinese but have not check yet)
    
    Parameters
    ----------
    prediction : TYPE string
        DESCRIPTION. sentence that predicted by the model
    ground_truth : TYPE string
        DESCRIPTION. raw sentence of the dataset

    Returns
    -------
    TYPE float
        DESCRIPTION. f1-score that does not count the order of character

    '''
    prediction_tokens = split_unicode_chrs(prediction)
    ground_truth_tokens = split_unicode_chrs(ground_truth)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    '''
    Extract match method that counted for the order of character in the sequence.
    the normalizer is only used for Japanese language

    Parameters
    ----------
    prediction : TYPE string
        DESCRIPTION. sentence that predicted by the model
    ground_truth : TYPE string
        DESCRIPTION. raw sentence of the dataset

    Returns
    -------
    TYPE bool
        DESCRIPTION. True or False

    '''
    # return (normalize_answer_jp(prediction) == normalize_answer_jp(ground_truth))
    return (neologdn.normalize(prediction) == neologdn.normalize(ground_truth))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    '''
    Get max score in the bow of True answer given in the dataset. 
    Equivalent to the Greedy method for choosing score

    Parameters
    ----------
    metric_fn : TYPE object function
        DESCRIPTION. function to calculate the score (F1-score or extract match method)
    prediction : TYPE string
        DESCRIPTION. sentence that predicted by the model
    ground_truth : TYPE string
        DESCRIPTION. raw sentence of the dataset

    Returns
    -------
    TYPE float
        DESCRIPTION. the highest score in the list

    '''
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                print('ground_truths: {}'.format(ground_truths))
                prediction = predictions[qa['id']]
                print('prediction: {}'.format(prediction))
                try:
                    exact_match += metric_max_over_ground_truths(
                        exact_match_score, prediction, ground_truths)
                    f1 += metric_max_over_ground_truths(
                        f1_score, prediction, ground_truths)
                    total += 1 # greedy method
                except:
                    print('the question {} has no ground truth'.format(qa['id']))

    exact_match = round(100.0 * exact_match / total, 2)
    f1 = round(100.0 * f1 / total, 2)

    return {'exact_match': exact_match, 'f1': f1}

#%%
'''
# Example of using evaluator
# Opening JSON file 
f = open(r'D:\VBDI_NLP\bert\tmp\squad_basecnn_jp\predictions.json') 
data_pred = json.load(f) 

f0 = open(r'D:\VBDI_NLP\jbddata\jbddata_dev_data.json', 'rb')
data = json.load(f0, encoding = 'utf-8')["data"]

test_eval = evaluate(data, data_pred)

# result: {'exact_match': 54.33, 'f1': 81.36}
s = '令和2年5月20日(水)14時00分~'
neologdn.normalize(s)
normalize_answer_jp(s)
'''