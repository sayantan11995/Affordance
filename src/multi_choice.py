import pandas as pd
import json
from io import BytesIO
import urllib
import os 
import torch
from tqdm import tqdm
import datasets
from transformers import BertTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup, BertConfig
from transformers import AutoModelForMaskedLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from transformers import pipeline
import numpy as np

from utils import eval


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

filename = "../data/ECCV_affordance_data.tsv"
filename = "../data/toloka_annotated_data.tsv"
filename = "../data/final_annotated_data.tsv"
# filename = "../data/Daivik_annotated.tsv"

data = pd.read_csv(filename, sep='\t')

# print(data.Sentence.loc[0])


label_names = list(data.columns[2:])
oov_class_map = {"lookthrough": "looking", "siton": "sitting", "pourfrom": "pouring", "writewith": "writing", "typeon": "typing"}

labels_lower = [x.lower() for x in label_names]
classes = []
for lab in labels_lower:
    if lab not in oov_class_map.keys():
        classes.append(lab)
    else:
        classes.append(oov_class_map[lab])



def init_model(model_name):
    if model_name == "roberta":
        model = pipeline("zero-shot-classification", model="roberta-large-mnli", device=0)

    elif model_name == "bart":
        model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
    elif model_name == "bert":
        model = pipeline("zero-shot-classification", model="sledz08/finetuned-bert-piqa", device=0)
    else:
        print("Error: model not supported!!!\nSupported NLI models: [roberta, bart]")

    return model

model = init_model('bart')

def predict_affordance_proba(model, sentence, object):
    word2scores = {}

    choices =  [object + " can be used for " + affordance + " by human"  for affordance in classes]
    choices = [affordance for affordance in classes]

    model_output = model(sentence, choices, multi_label=True)
    labels = model_output['labels']
    scores = model_output['scores']
    option2scores = {labels[i]: scores[i] for i in range(len(choices))}
    pred_scores = [option2scores[option] for option in choices]

    # print(choices)
    # print(word2scores)
    
    # print(option2scores)
    # print(word2scores)

    for lab, prob in zip(classes, pred_scores):
        word2scores[lab] = prob

    return word2scores


## Initialize to calculate Accuracy
gt_affordance = {}
correct = 0
wrong = 0

## Initialize to calculate Mean Average Precision
ground_truth_classes = []
predicted_classes = []


for ids, rows in data.iterrows():
# predicted_affordances = predict_affordance_proba(model, sentence, object)
    # if ids >=200:
    #     break
    positive_classes = []
    negative_classes = []
    sentence = rows[0]
    object = rows[1]

    predicted_affordances = predict_affordance_proba(model, sentence, object)
    sorted_predicted_affordances = dict(sorted(predicted_affordances.items(), key=lambda item: item[1], reverse=True))

    for itr in range(2, data.shape[1]):
        class_name = list(data.columns)[itr].lower()
        ## replacing oov classnames
        class_name = class_name if class_name not in oov_class_map.keys() else oov_class_map[class_name]

        gt_affordance[class_name] = rows[itr]

        if rows[itr] > 0:
            positive_classes.append(class_name)
        else:
            negative_classes.append(class_name)

    prev_cor = correct
    prev_wrong = wrong
    
    if len(positive_classes) > 0:
        for pos in positive_classes:
            for neg in negative_classes:
                if predicted_affordances[pos] >= predicted_affordances[neg]:
                    correct += 1
                else:
                    wrong += 1

        print((correct-prev_cor)/(correct+wrong - prev_cor-prev_wrong))

        ground_truth_classes.append(positive_classes)
        predicted_classes.append(list(sorted_predicted_affordances.keys()))
        # print(sorted_predicted_affordances)
        # print(positive_classes)

    # if ids > 5:
    #     break

accuracy = correct / (correct+wrong)

print("Accuracy: %s"%accuracy )
print("MAP: %s" %eval.mapk(ground_truth_classes, predicted_classes, 15))