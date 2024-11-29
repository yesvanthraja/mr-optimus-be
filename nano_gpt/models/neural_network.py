import torch 
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        """
        The Bigram Language Model.
        This would create a vocab_size x vocab_size matrix where the index after passing would be reffering to the extract row column in the matrix.
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, index, targets):
        logits = self.token_embedding_table(index)  # now pytorch will arrange the data in Batch = 4 by Times = 8(block_size) by Channels = 65(characters)(B,T,C) torch.Size([4, 8, 65])
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets) # how well are we predicting the next token
        return logits, loss
    
    def generate(self, index, mex_new_tokens):
        pass
    
