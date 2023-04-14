import torch
from torch import nn

class Linearbert(nn.Module):
  def __init__(self, bert_model, hidden_dim=384, dropout=0.1):
    super(Linearbert, self).__init__()

    self.bert = bert_model
    self.lin1 = nn.Linear(768, hidden_dim)
    self.lin2 = nn.Linear(hidden_dim, 7)
    if dropout:
          self.dropout = nn.Dropout(p=dropout)


  # BERT input 형태: input_ids, token_type_ids, attention_masks
  def forward(self, token_ids, segment_ids, attention_mask):
    _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), 
                            attention_mask = attention_mask.float(), return_dict=False) # (batch, 768)
    dropout_o = self.dropout(pooler)
    o = self.lin1(dropout_o)
    o = self.lin2(o)
    
    return o # (batch, 7)