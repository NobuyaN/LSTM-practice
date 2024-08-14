import torch 
import torch.nn as nn

"""
    CREDITS
    ____________
    The design is proposed by a group of researcher, Vaswani et al, at Google
    from the paper https://arxiv.org/pdf/1706.03762. 
"""

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int): # dimension of the embedding, vocabulary size of the model (character-level, token-level, etc)
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * torch.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float): # dimension of the positional encoding must equal dim of input embedding (typically 512), and the length of sentence
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        """
            Positional Encoding Notes
            _________________________
            Even Dimension 2i: torch.sin(pos/(10000 ** ((2 * i) / d_model)))
            Odd dimension 2i+1: torch.cos(pos/(10000 ** ((2 * i) / d_model))) | The positional encoding of 2i+1 should be 2i of pe with cos
            let pos/(10000 ** ((2 * i) / d_model)) similarily be computed to reduce redundancy
            Logarithm can be used for numerical stability, but I have omitted it
        """

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        denominator = torch.pow(10000, 2 * torch.arange(0, d_model, 2, dtype=torch.float) / d_model) # This array would have length of d_model / 2 

        #Add the positional encoding for sin (even dimension index) (2i)
        # Fills up variable pe with shape of (seq_len, d_model / 2)
        pe[:, 0::2] = torch.sin(position * denominator)

        #Then for cos (odd) (2i+1)
        # Fills up vairable pe with the leftover from the sin
        pe[:, 1::2] = torch.cos(position * denominator) 

        # Set the variable pe as each batch
        pe = pe.unsqueeze(0) # Shape (1, seq_len, d_model)

        # Register pe in the buffer, allowing self.pe as it is stored in module's state
        # Accessible with instance_name.state_dict().keys()
        # Automatically creates attribute of self.pe
        self.register_buffer("pe", pe) 

    def forward(self, x):
        # x.shape[1] to ensure that x and self.pe is similar in dimension
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, eps=10e-6):
        super().__init__()
        self.eps = eps

        #alpha and gamma is a learnable parameter (multiplicative and additive respectively)
        self.gamma = nn.Parameter(torch.ones(1)) 
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdims=True)
        std = x.std(dim=-1, keepdims=True)
        # new_x = x - mean / sqrt(std ** 2 + eps)
        # std ** 2 is variance, we can remove the square root for easier computation 
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # Bias is already defined, defaulted as True in nn.Linear()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()

        """
        Create weight matrix of (d_model, d_model) and then split it into h heads
        concatenate the multi head later after attention computation

        compute with the attention formula
            Attention(Q, K, V) = softmax(Q * K_T / sqrt(d_k)) * V 

        In the original attention paper, they used 8 heads, yielding 64 embedding size vector d_k (512 / 8)
        """
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h, not resulting to whole number"
        self.d_k = d_model / h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod # Allows no instantiation. Can use the class name.
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # query and key dimension (batch, h, seq_len, d_k) 
        # key transposed (-2, -1) --> (batch, h, d_k, seq_len)
        # batch and h stays the same. Matrix multiplication of (seq_len, d_k) to (d_k, seq_len) resulting in (seq_len, seq_len)
        # attention_scores = query @ key transposed --> (batch, h, seq_len, seq_len)
        attention_scores = query @ key.transpose(-2, -1) / torch.sqrt(query.shape[-1])
        if mask is not None:
            # Replace value with -1e9 for position in which the value in mask is 0, correspoding to attention_scores matrix
            # e^(-1e9) roughly equals 0, masking with auto-regressive nature
            attention_scores.masked_fill_(mask==0, "-1e9")
        # Apply softmax in the last dimension of attention_scores with porbability values (batch, h, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim=-1) 
        if dropout is not None:
            attention_scores = dropout(attention_scores) # Same as doing nn.Dropout(n: float)(input)
        
        return (attention_scores @ value), attention_scores 
        
    def forward(self, q, k, v, mask):
        # Using the forward method of the nn.Linear class, pass in the input tensor
        # Allwoing matrix multiplication of (input X weights) in the respective order
        query = self.w_q(q) 
        key = self.w_k(k)
        value = self.w_v(v)

        # Original query dimension (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        # d_model = self.h * self.d_k
        # Let h oversee self.d_k and seq_len
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) 
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2) 
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2) 

        x, self.attention_scores = MultiHeadedAttention.attention(query, key, value, mask, self.dropout)

        # x dimension --> (batch, h, seq_len, d_k)
        # x.transpose(1,2) --> (batch, seq_len, h, d_k)
        x = x.transpose(1, 2)
        # x.view --> (batch, seq_len, d_model)
        x = x.view(x.shape[0], x.shape[1], self.d_model)
        # x @ w_o = (batch, seq_len, d_model) @ (d_model, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)
        
