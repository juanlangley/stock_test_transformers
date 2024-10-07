import torch
import torch.nn as nn

class SelfAttention(nn.Module):
  def __init__(self, embed_size, heads=8):
    super(SelfAttention, self).__init__()
    self.embed_size = embed_size
    self.heads = heads
    self.head_dim = embed_size // heads
    assert self.head_dim * heads == embed_size, "embed_size must be divisible by heads"

    self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

  def forward(self, values, keys, query, mask):
    N = query.shape[0]

    value_len = values.shape[1]
    key_len = keys.shape[1]
    query_len = query.shape[1]

    print(values)
    print(value_len, values.shape[1], values.shape)
    print(N, self.heads, self.head_dim)

    values = values.reshape(N, value_len, self.heads, self.head_dim)
    keys = keys.reshape(N, key_len, self.heads, self.head_dim)
    queries = query.reshape(N, query_len, self.heads, self.head_dim)

    values = self.values(values)
    keys = self.keys(keys)
    queries = self.queries(queries)

    #Q*K transpuesta - energy
    energy = torch.einsum("nqhd, nkhd -> nhqk", [queries, keys])
    # queries shape: (N, query_len, heads, head_dim)
    # keys shape: (N, key_len, heads, head_dim)
    # energy shape: (N, heads, query_len, key_len)
    # query_len es la oracion de destino y kel_len la oracion fuente - para cada palabra objetivo cuanto debemos prestar antención a cada entrada

    if mask is not None:
      energy = energy.masked_fill(mask == 0, float("-1e20"))
      attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
    else:
      attention = torch.softmax(energy, dim=3)
    #l = dimensión que queremos multiplicar
    out = torch.einsum("nhql, nlhd -> nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
    #attention shape: (N, heads, query_len, key_len)
    #Values shape: (N, values_len, head, head_dim)
    #Out (lo que queremos): (N, query_len, heads, head_dim)
    # after out (einsum) aplanamos las ultimas dos dimensiones (flatten heads heads_dim)
    out = self.fc_out(out)
    return out

# Multi-Head Attention

class TransformerBlock(nn.Module):
  def __init__(self,
              embed_size,
              heads,
              dropout,
              forward_expansion):
    super(TransformerBlock, self).__init__()
    self.attention = SelfAttention(embed_size, heads)
    self.norm1 = nn.LayerNorm(embed_size)
    self.norm2 = nn.LayerNorm(embed_size)
    self.feed_forward = nn.Sequential(
        nn.Linear(embed_size, forward_expansion * embed_size),
        nn.ReLU(),
        nn.Linear(forward_expansion * embed_size, embed_size))
    self.dropout = nn.Dropout(dropout)

  def forward(self, value, key, query, mask):
    attention = self.attention(value, key, query, mask)
    x = self.norm1(attention + query)
    x = self.dropout(x)
    forward = self.feed_forward(x)
    out = self.norm2(forward + x)
    out = self.dropout(out)
    return out
  
class Encoder(nn.Module):
  def __init__(self,
               src_vocab_size,
               embed_size,
               num_layers,
               heads,
               device,
               forward_expansion,
               dropout,
               max_length):
    #max_length relacionado con incrustación posicional (posición correcta)
    super(Encoder, self).__init__()
    self.embed_size = embed_size
    self.device = device
    self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
    self.position_embedding = nn.Embedding(max_length, embed_size)
    self.layers = nn.ModuleList([
        TransformerBlock(
            embed_size,
            heads,
            dropout,
            forward_expansion) for _ in range(num_layers)
        ])
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask):
    N, seq_len = x.shape
    positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
    self.word_embedding(x) + self.position_embedding(positions)
    x = self.dropout(x)
    for layer in self.layers:
      x = layer(x, x, x, mask)
    return x
  
class DecoderBlock(nn.Module):
  def __init__(self,
               embed_size,
               heads,
               forward_expansion,
               dropout,
               device):
    super(DecoderBlock, self).__init__()

    self.norm = nn.LayerNorm(embed_size)
    self.attention = SelfAttention(embed_size, heads)
    self.transformer_block = TransformerBlock(
        embed_size,
        heads,
        dropout,
        forward_expansion)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, value, key, src_mask, trg_mask):
    attention = self.attention(x, x, x, trg_mask)
    query = self.dropout(self.norm(attention + x))
    out = self.transformer_block(value, key, query, src_mask)
    return out
  
class Decoder(nn.Module):
  def __init__(self,
               trg_vocab_size,
               embed_size,
               num_layers,
               heads,
               forward_expansion,
               dropout,
               device,
               max_length):
    super(Decoder, self).__init__()
    self.device = device
    self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
    self.position_embedding = nn.Embedding(max_length, embed_size)

    self.layers = nn.ModuleList([
        DecoderBlock(
            embed_size,
            heads,
            forward_expansion,
            dropout,
            device) for _ in range(num_layers)
            ])
    self.fc_out = nn.Linear(embed_size, trg_vocab_size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, enc_out, src_mask, trg_mask):
    N, seq_len = x.shape
    positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
    x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))
    for layer in self.layers:
      x = layer(x,
                enc_out,
                enc_out,
                src_mask,
                trg_mask)
    out = self.fc_out(x)
    return out
  
class Transformer(nn.Module):
  def __init__(self,
               src_vocab_size,
               trg_vocab_size,
               src_pad_idx,
               trg_pad_idx,
               embed_size=256,
               num_layers=6,
               forward_expansion=4,
               heads=8,
               dropout=0,
               device="cuda" if torch.cuda.is_available() else "cpu",
               max_length=100):

               #El pad es pq tienen q tener la misma longitud

               super(Transformer, self).__init__()
               self.encoder = Encoder(
                   src_vocab_size,
                   embed_size,
                   num_layers,
                   heads,
                   device,
                   forward_expansion,
                   dropout,
                   max_length)
               self.decoder = Decoder(
                   trg_vocab_size,
                   embed_size,
                   num_layers,
                   heads,
                   forward_expansion,
                   dropout,
                   device,
                   max_length)

               self.src_pad_idx = src_pad_idx
               self.trg_pad_idx = trg_pad_idx
               self.device = device
  def make_src_mask(self, src):
    src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
    # (N, 1, 1, src_len)
    return src_mask.to(self.device)

  def make_trg_mask(self, trg):
    N, trg_len = trg.shape
    trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
    return trg_mask.to(self.device)

  def forward(self, src, trg):
    src_mask = self.make_src_mask(src)
    trg_mask = self.make_trg_mask(trg)
    enc_src = self.encoder(src, src_mask)
    out = self.decoder(trg, enc_src, src_mask, trg_mask)
    return out
  

# EJEMPLO

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  x = torch.tensor([[1,5,6,4,3,9,5,2,0], [1,8,7,3,4,5,6,7,2]]).to(device)
  trg = torch.tensor([[1,7,4,3,5,9,2,3,0], [1,5,6,2,4,1,7,6,2],[4,2,1,5,6,7,4,3,1]]).to(device)

  src_pad_idx = 0
  trg_pad_idx = 0
  src_vocab_size = 10
  trg_vocab_size = 10
  model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
  # Sacamos el último dato pq es el que queremos predecir
  out = model(x, trg[:, :-1])
  print(out.shape)

"""
uncorrected error

<ipython-input-41-116bbcf5a603> in forward(self, values, keys, query, mask)
     27     print(f"query shape: {query.shape}")
     28 
---> 29     values = values.reshape(N, value_len, self.heads, self.head_dim)
     30     keys = keys.reshape(N, key_len, self.heads, self.head_dim)
     31     queries = query.reshape(N, query_len, self.heads, self.head_dim)

RuntimeError: shape '[2, 9, 8, 32]' is invalid for input of size 18
"""