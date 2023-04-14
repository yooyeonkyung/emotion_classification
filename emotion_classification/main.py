import torch
from transformers import BertModel
from torch.utils.data import DataLoader
from kobert_tokenizer import KoBERTTokenizer
from data.dataset import Sentiment
from core.model import Linearbert
from utils.wrapper import train_model
from utils.wrapper_1 import train_model_1

import gluonnlp as nlp
import pandas as pd
import numpy as np
import random
from datetime import datetime
import argparse

# random seed 고정
RANDOM_SEED = 11
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

# GPU 사용
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# KoBERT
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bert_model = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
for param in bert_model.parameters(): # KoBERT freeze
    param.requires_grad=False
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

def main(configs):
    
    # curriculum data
    train_data = pd.read_csv('./data/curriculum/kem20_h2l_0.csv')
    
    # train/ test data
    # train_data = pd.read_csv('./data/KEMDy20/kem20_tr0.csv')
    test_data = pd.read_csv('./data/KEMDy20/kem20_te0.csv')
    
    print(len(train_data))
    print(len(test_data))

    tr_data = Sentiment(train_data)
    te_data = Sentiment(test_data)

    train_dataloader = DataLoader(tr_data, configs.batch_size, shuffle=configs.shuffle) # curriculum 경우: False
    test_dataloader = DataLoader(te_data, configs.batch_size, shuffle=False)

    if configs.mode == "train":
        if train_data is None:
            raise ValueError(f"'--train_file' expected '*.csv', got '{configs.TRAIN_FILE}'")
        
        start_time = datetime.now()
        print(f"\n[START] {start_time:%Y-%m-%d @ %H:%M:%S}")

        model = Linearbert(bert_model)
        model = model.to(device)
        
        if configs.th == False:
            train_model(model, train_dataloader, test_dataloader, configs.epochs)
        else: train_model_1(model, train_dataloader, test_dataloader, configs.epochs)

        finish_time = datetime.now()
        print(f"\n[FINISH] {finish_time:%Y-%m-%d @ %H:%M:%S} (user time: {finish_time - start_time})\n")
    



def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=False,
        default='train',
        help="mode - train or not",
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="batch size which is used (default - 8)",
    )
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="epoch which is used in train mode (default - 10)",
    )
    parser.add_argument(
        "--shuffle",
        default=True,
        type=bool,
        help="whether training data shuffle is true or false (default - True)"
    )
    parser.add_argument(
        "--th",
        default=False,
        type=bool,
        help="whether to use new threshold (defalut - False)"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    configs = parse_arguments()
    main(configs)
