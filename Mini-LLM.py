
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- 1. Tiny dataset ----
from datasets import load_dataset

# load dataset
dataset = load_dataset("roneneldan/TinyStories", split="train")

dataset = dataset.select(range(1000))

# combine into one string
text = " ".join(dataset["text"])

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long).to(device)

# ---- 2. Batching ----
block_size = 64

def get_batch():
    start = random.randint(0, len(data) - block_size - 1)
    x = data[start:start+block_size]
    y = data[start+1:start+block_size+1]
    return x, y

# ---- 3. Model ----
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        seq_len = x.shape[0]

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # split into heads
        Q = Q.view(seq_len, self.num_heads, self.head_dim)
        K = K.view(seq_len, self.num_heads, self.head_dim)
        V = V.view(seq_len, self.num_heads, self.head_dim)

        # transpose for attention
        Q = Q.transpose(0,1)  # (heads, seq, dim)
        K = K.transpose(0,1)
        V = V.transpose(0,1)

        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)

        mask = torch.tril(torch.ones(scores.size(-2), scores.size(-1), device=x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)

        out = weights @ V  # (heads, seq, dim)

        # combine heads
        out = out.transpose(0,1).contiguous().view(seq_len, -1)

        return self.fc_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.attn = MultiHeadAttention(embed_size, num_heads=4)
        self.ln1 = nn.LayerNorm(embed_size)

        self.ff = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size)
        )
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # attention + residual
        x = x + self.ff(self.ln2(x))     # feedforward + residual
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, embed_size=128, max_len=100):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(max_len, embed_size)

        self.blocks = nn.Sequential(
    TransformerBlock(embed_size),
    TransformerBlock(embed_size),
    TransformerBlock(embed_size),
    TransformerBlock(embed_size),
)

        self.ln = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        seq_len = x.size(0)
        positions = torch.arange(0, seq_len, device=x.device)

        x = self.embed(x) + self.pos_embed(positions)
        x = self.blocks(x)

        x = self.ln(x)
        logits = self.head(x)
        return logits


model = TinyGPT(vocab_size).to(device)

# ---- 4. Training ----
optimizer = optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.CrossEntropyLoss()

for step in range(4000):

    x, y = get_batch()

    logits = model(x)
    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, Loss: {loss.item()}")

# ---- 5. Generate ----
model.eval()

x = torch.tensor([data[0].item()], device=device)

for _ in range(60):
    logits = model(x)

    temperature = 0.9
    #probs = torch.softmax(logits[-1] / temperature, dim=0)

    #next_token = torch.multinomial(probs, 1)
    k = 5

    logits = logits[-1] / temperature

    values, indices = torch.topk(logits, k)
    probs = torch.softmax(values, dim=0)
    next_token = indices[torch.multinomial(probs, 1)]
    x = torch.cat([x, next_token])

print("Generated:", decode(x.tolist()))