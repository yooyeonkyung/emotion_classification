from transformers import BertModel
from torch.utils.data import DataLoader
from kobert_tokenizer import KoBERTTokenizer
from data.dataset import Sentiment
from core.model import Linearbert\

import torch
import gluonnlp as nlp
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

# random seed 고정
RANDOM_SEED = 11
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

# GPU 사용
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# KoBERT
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bert_model = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

model = Linearbert(bert_model)
model.load_state_dict(torch.load('./saved/230413_10/230413_10_ep5_model_save.pt', map_location='cuda:1'))
model = model.to(device)
kemdy20 = pd.read_csv('./data/KEMDy20/kem20_tr0.csv')

dataset = Sentiment(kemdy20)
test_dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

def validation(model, test_dataloader):
    model.eval()

    with torch.no_grad(): 
        final_score = []     
        for (token_ids, segment_ids, attention_mask, label) in tqdm(test_dataloader):
            # data to gpu
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            attention_mask = attention_mask.long().to(device)
            label = label.long().to(device)            

            out = model(token_ids, segment_ids, attention_mask)
            out = torch.sigmoid(out)
            
            for i in range(len(label)):
                score = []
                for j in range(len(label[i])):
                    if label[i][j] == 0:
                        score.append(out[i][j].item())
                    else:
                        score.append(1-out[i][j].item())
                final_score.append(sum(score))
            
        return final_score

final_score = validation(model, test_dataloader)

dataframe = pd.DataFrame(final_score)
dataframe.to_csv('./data/curriculum/score_0.csv')