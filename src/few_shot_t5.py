
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

set_seed(42)

model_name = 'google/flan-t5-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(model_name)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
t5_model.to(device)

# optimizer
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in t5_model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in t5_model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr=3e-4, eps=1e-8)


# dataset preparation

true_false_adjective_tuples = [
                               ("Historians were able to decipher ancient Egyptian hieroglyphics by studying the inscriptions on the Rosetta Stone. Can human watch Rosetta Stone?","Yes"),
                               ("Historians were able to decipher ancient Egyptian hieroglyphics by studying the inscriptions on the Rosetta Stone. Can human feed Rosetta Stone?","No"),
                               ("With his saz in hand, the musician strummed the strings, filling the air with the melodic sounds of Turkish folk music. Can human play with  saz?","Yes"),
                               ("With his saz in hand, the musician strummed the strings, filling the air with the melodic sounds of Turkish folk music. Can human sit on saz?","No"),
                               ("With the dovetail jig, I can create strong and beautiful dovetail joints for my woodworking projects. Can human lift dovetail jig?","Yes"),
                               ("With the dovetail jig, I can create strong and beautiful dovetail joints for my woodworking projects. Can human ride with dovetail jig?","No"),
                               ("The Rosetta Probe played a vital role in unlocking the mysteries of comet 67P/Churyumov-Gerasimenko. Can human fix Rosetta Probe?","Yes"),
                               ("The Rosetta Probe played a vital role in unlocking the mysteries of comet 67P/Churyumov-Gerasimenko. Can human play Rosetta Probe?","No"),
                               ("In her flowing hanfu gown, she gracefully walked through the ancient palace, embodying the elegance and traditions of Chinese culture. Can human grasp hanfu?","Yes"),
                               ("In her flowing hanfu gown, she gracefully walked through the ancient palace, embodying the elegance and traditions of Chinese culture. Can human look through hanfu?","No")
]

t5_model.train()

epochs = 4

for epoch in range(epochs):
  print ("epoch ",epoch)
  for input,output in true_false_adjective_tuples:
    input_sent = "Answer Yes or No: "+input
    ouput_sent = output

    tokenized_inp = tokenizer.encode_plus(input_sent,  max_length=96, pad_to_max_length=True,return_tensors="pt")
    tokenized_output = tokenizer.encode_plus(ouput_sent, max_length=96, pad_to_max_length=True,return_tensors="pt")


    input_ids  = tokenized_inp["input_ids"].to(device)
    attention_mask = tokenized_inp["attention_mask"].to(device)

    lm_labels= tokenized_output["input_ids"].to(device)
    decoder_attention_mask=  tokenized_output["attention_mask"].to(device)


    # the forward function automatically creates the correct decoder_input_ids
    output = t5_model(input_ids=input_ids, labels=lm_labels, decoder_attention_mask=decoder_attention_mask,attention_mask=attention_mask)
    loss = output[0]

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

test_sent = 'Answer Yes or No: The metallic jingles of the sistrum added a mesmerizing rhythm to the ancient Egyptian ritual, invoking the divine presence.	Can human play Sistrum? </s>'
test_sent = "Answer Yes or No: The flowing chador enveloped her in modesty and grace, representing the cultural values and religious customs of Iran.	Can human feed Chador?"

test_tokenized = tokenizer.encode_plus(test_sent, return_tensors="pt")

test_input_ids  = test_tokenized["input_ids"]
test_attention_mask = test_tokenized["attention_mask"]

t5_model.eval()
beam_outputs = t5_model.generate(
    input_ids=test_input_ids,attention_mask=test_attention_mask,
    max_length=20,
    early_stopping=True,
    num_beams=4,
    num_return_sequences=3,
    no_repeat_ngram_size=2
)

for beam_output in beam_outputs:
    sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    print (sent)


