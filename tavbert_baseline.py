import random
import re
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, TrainingArguments, Trainer
from dataset import textDataset, NIQQUD_SIZE, DAGESH_SIZE, SIN_SIZE
from torch import nn
import numpy as np
import sklearn


tokenizer = AutoTokenizer.from_pretrained("tavbert")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForMaskedLM.from_pretrained("tavbert")
with open(Path.cwd() / r"hebrew_diacritized/data/test/books/2.txt", encoding='utf8') as f:
    res = open("res.txt",'w',encoding='utf8')
    for sent in f.readlines():
        word = random.choice(sent.split(" "))
        res.write(f"the word chosen is : {word}\n")

        word_sub = re.sub("[^א-ת]", "", word)
        res.write(f"the word without nikkud : {word_sub}\n")
        inp = sent.replace(word, "[MASK]".join(word_sub))
        res.write(f"model input : {inp}\n")
        toks = tokenizer.encode(inp, return_tensors="pt")
        pridiction = model(toks)
        p = torch.argmax(pridiction.logits, dim=-1)
        outp = tokenizer.decode(p.squeeze(), skip_special_tokens=True, return_offsets_mapping=True)
        # res.write(outp[::2])
        res.write(f"model output : {outp[::2]}\n\n")
        res.write(f"model all : {outp}\n\n")
    res.close()
        #     t = tokenizer.encode(,return_tensors="pt")
        #     print(":".join(data["text"]))
        #     p = torch.argmax(model(t).logits, dim=-1)
        #     print(tokenizer.decode(p.squeeze()))
        # # for i in range(100, 200):
        # #     print(tokenizer.decode(p[i]))

    #     t = tokenizer.encode(,return_tensors="pt")
    #     print(":".join(data["text"]))
    #     p = torch.argmax(model(t).logits, dim=-1)
    #     print(tokenizer.decode(p.squeeze()))
    # # for i in range(100, 200):
    # #     print(tokenizer.decode(p[i]))

# pred = trainer.predict(small_eval_dataset)
# for data in iter(small_eval_dataset):
#     sent = data["text"]
#     word = random.choice(sent.split(" "))
#     print(f"the word chosen is : {word}")
#     masked_word = "[MASK][MASK][MASK]".join(word)
#     t = tokenizer.encode(,return_tensors="pt")
#     print(":".join(data["text"]))
#     p = torch.argmax(model(t).logits, dim=-1)
#     print(tokenizer.decode(p.squeeze()))
# # for i in range(100, 200):
# #     print(tokenizer.decode(p[i]))
