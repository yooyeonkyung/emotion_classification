import torch
from kobert_tokenizer import KoBERTTokenizer
from torch.utils.data import Dataset

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

# label: neutral, happy, surprise, angry, disgust, sad, fear

def resampling_data(df):
    inputs = tokenizer(df['data'].values.tolist(), return_tensors="pt", padding=True, truncation=True, max_length=256)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    token_type_ids = inputs.token_type_ids
    x = [[input_id, token_id, mask] for input_id, token_id, mask in zip(input_ids, token_type_ids, attention_mask)]
    y = [[neu, hap, sur, fea, ang, dis, sad] for neu, hap, sur, fea, ang, dis, sad in zip(df['neutral'], df['happy'], df['surprise'], df['fear'], df['angry'], df['disgust'], df['sad'])]
    return x, y

class Sentiment(Dataset):
    def __init__(self, dataset):
        X, y = resampling_data(dataset)
        
        self.sentences = []
        self.attention_masks = []
        self.token_ids = []
        for input_ids, token_ids, masks in X:
            self.sentences.append(input_ids)
            self.token_ids.append(token_ids)
            self.attention_masks.append(masks)
        
        self.labels = torch.tensor(y)

    def __getitem__(self, i):
        return (self.sentences[i], self.token_ids[i], self.attention_masks[i], self.labels[i])

    def __len__(self):
        return (len(self.labels))