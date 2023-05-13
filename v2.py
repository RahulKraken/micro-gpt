import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# hyperparameters
batch_size = 64 # independent sequences fed in one pass
block_size = 256 # context length
max_iters = 5_000
eval_interval = 500
lr = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
device = 'cpu' #'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"device: {device}")

torch.manual_seed(1332) # for reproducibility

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
text = open('input.txt', 'r').read()
start = '\n'

# build vocab
tokens = sorted(list(set(''.join(text))))
# let's assume '#' is my delimiter (start, end) token

# we need to embed these tokens into some numbers
stoi = {x: i + 1 for i, x in enumerate(tokens)}
stoi['#'] = 0
itos = {i: x for x, i in stoi.items()}
vocab_size = len(itos)

# character level tokenizer
# TODO: try sub-word tokenizer
encode = lambda s: [stoi[c] for c in s] # string -> vector
decode = lambda x: ''.join([itos[i] for i in x]) # vector -> string

# encode the dataset
data = torch.tensor(encode(text), dtype=torch.long)

# build the dataset
n = int(0.9 * len(data))
d_train = data[:n]
d_val = data[n:]

def get_batch(split):
    # gen batch of random x and y
    data = d_train if split == 'train' else d_val
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = m(x, y)
            losses[k] = loss
        out[split] = losses.mean()
    m.train()
    return out

# single head self-attention
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # eval affinity
        wei = q @ k.transpose(-2, -1) * C ** 0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

# multi head self-attention
class MultiHead(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size=head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) 
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

# attention block
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHead(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# simple bigram
class BigramLLM(nn.Module):
    def __init__(self):
        super().__init__()
        # each token reads logits for the next token from lookup table
        self.token_emb_table = nn.Embedding(vocab_size, n_embd)
        self.pos_emb_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd=n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are (B, T)
        # B -> batch dim
        # T -> time dim -> ctx
        # after embedding them, each idx[i, j] will be have emb dim
        # it'll become (B, T, C) -> C is emb dim
        t_embd = self.token_emb_table(idx)
        p_embd = self.pos_emb_table(torch.arange(T, device=device)) # (T, C)
        x = t_embd + p_embd
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # -ve log liklihood loss
            # cross_entropy in pytorch
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of current ctx indices
        for _ in range(max_new_tokens):
            # crop idx to block_size
            idx_cond = idx[:, -block_size:]
            # predictions
            logits, loss = self(idx_cond)
            # focus on latest time step
            logits = logits[:, -1, :] # pick latest T from (B, T, C) -> becomes (B, C)
            # softmax
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # add to running index
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLLM()
m.to(device)

# pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    # eval loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print('\n\n\n\n')
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
print(decode(m.generate(x, 1000)[0].tolist()))
