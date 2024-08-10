import torch 
import torch.nn as nn
import math

"""
    CREDITS
    ____________
    The design is proposed by a group of researcher, Vaswani et al, at Google
    from the paper https://arxiv.org/pdf/1706.03762. 
"""

class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size): # dimension of the embedding, vocabulary size of the model (character-level, token-level, etc)
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout: float): # dimension of the positional encoding must equal dim of input embedding (typically 512), and the length of sentence
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Even dimension 2i: math.sin(pos/(10000 ** ((2 * i) / d_model)))
        # Odd dimension 2i+1: math.cos(pos/(10000 ** ((2 * i) / d_model)))

        pe = torch.zeros(seq_len, d_model)

        for pos in range(seq_len):
            for i in range(d_model): 
                # let pos be the positional index of the word in the sentence and i be the index of the dimension 
                if i % 2 == 0:
                    pe[pos, i] = math.sin(pos / math.pow(10000, (2 * i) / d_model))
                else:
                    pe[pos, i] = math.cos(pos / math.pow(10000, (2 * i) / d_model))
    
class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdims=True)
        std = x.std(dim=-1, keepdims=True)
        return self.alpha * (x - mean) / (torch.sqrt(torch.pow(std, 2) + self.eps)) + self.bias

class MultiHeadedAttention(nn.Module):
    pass

class 

