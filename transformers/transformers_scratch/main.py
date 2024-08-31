import torch 
import torch.nn as nn

"""
    CREDITS
    ____________
    The design is proposed by a group of researcher, Vaswani et al, at Google
    from the paper https://arxiv.org/pdf/1706.03762. 
"""

class InputEmbeddings(nn.Module):
    # Create look-up table of numerical representation of tokens inside a sentence
    def __init__(self, d_model: int, vocab_size: int): # dimension of the embedding, vocabulary size of the model (character-level, token-level, etc)
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # x would be the indices of the tokens, which would return the respective dense vectors that hold informations
        # After tokenization, the list of words in the sequence would be converted into indices with the vocab_size
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

class MultiHeadedAttentionBlock(nn.Module):
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

        x, self.attention_scores = MultiHeadedAttentionBlock.attention(query, key, value, mask, self.dropout)

        # x dimension --> (batch, h, seq_len, d_k)
        # x.transpose(1,2) --> (batch, seq_len, h, d_k)
        x = x.transpose(1, 2).contiguous()
        # x.view --> (batch, seq_len, d_model)
        x = x.view(x.shape[0], x.shape[1], self.d_model)
        # x @ w_o = (batch, seq_len, d_model) @ (d_model, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)
        
class ResidualConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    # Define the function with x as tensor and sublayer as a function
    def forward(self, x: torch.Tensor, sublayer: function):
        # x + Transformed(x)
        # sublayer is nn.Linear with inputs of self.norm(x)
        # y = self.norm(x) @ W + b
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadedAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout(dropout)
        # Create 2 residual connection module with dropout in module list
        self.residual_connection = nn.ModuleList(ResidualConnection(dropout) for _ in range(2))

    def forward(self, x, src_mask):
        # First residual_connection module indexing
        # where argument sublayer = lambda x: self.self_attention_block(x, x, x, src_mask)
        # Seen from the forward function in residual_connection, sublayer is called with an input of self.norm(x)
        # therefore, sublayer(self.norm(x)) is called in residual_connection properly [callable function]
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # Second residual_connection module indexing
        # pass in a callable function, self.feed_forward_block, into the second parameter for residual_connection
        x = self.residual_connection[1](x, self.feed_forward_block)

        return x
    
class Encoder(nn.Module):
    # layer defined as a list of Modules 
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    # By using the required EncoderBlock positional argument (x, src_mask)
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
        
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadedAttentionBlock, cross_attention_block: MultiHeadedAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout(dropout)
        self.residual_connection = nn.ModuleList(ResidualConnection(dropout) for _ in range(3))
    
    # pass in encoder output to be fed into the multi headed attention
    # pass in the source mask (src_mask) which masks the encoder's output (Allows to disable attention mechanism of padding tokens)
        # padding tokens are zeroed out with the mask
    # pass in the target mask (tgt_mask) which masks the decoder (Allows prediction without oversight of future words in a sentence)
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # Query is from the decoder, while the keys and values is from the encoder
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
            return self.norm(x) 
        
class Projectionlayer(nn.Module):
    """
        This layer maps the (batch, seq_len, d_model) to a new dimension (batch, seq_len, vocab_size)
        High dimensional representation of decoder's output to the vocabulary space
        Ensures which words has the highest probability, which would be trained, by using log softmax for numerical stability

        The row, seq_len, would have column size of vocab_size. The softmax would output probability and the token with the
        highest probability would be the prediction of the next token. 
    """
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # softmax over last dimension (vocab_size), with probability sum of 1 for each row
        return torch.log_softmax(self.proj(x), dim=-1)
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: Projectionlayer):
        super().__init__()
        """
            Different embeddings for source and target, especially for translation task in which two languages would be better equipped with different representations
            Source Embedding: Converting input tokens into dense vectors 
            Target Embedding: Converting previously generated output tokens into dense vectors, predicting the consecutive tokens

            Different positional encodings. Source and target both needs to know the relative position inside the sentence. h

        """
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # src is defined to be a tensor of indices from the input tokens
        # Turn the src (tensor of indices) to tensor of dense vectors retaining embeddings
        src = self.src_embed(src)
        # Apply positional encoding to the src (tensor of dense vectors) 
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

# Create base transformer model    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, H: int = 8, dropout: float = 0.1,  d_ff: int = 2048):
    # Embedding layer
    src_embed  = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed  = InputEmbeddings(d_model, tgt_vocab_size)

    # Positional Encoding Layer
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Encoder and decoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention = MultiHeadedAttentionBlock(d_model, H, dropout)
        feed_foward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention, feed_foward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention = MultiHeadedAttentionBlock(d_model, H, dropout)
        decoder_cross_attention = MultiHeadedAttentionBlock(d_model, H, dropout)
        feed_foward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention, decoder_cross_attention, feed_foward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Encoder adn Decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Projection Layer
    projection_layer = Projectionlayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initializing parameter is a good habit, but I am not well equipped in this topic so I have omitted it 
    # Usually, xavier transform is used

    return transformer