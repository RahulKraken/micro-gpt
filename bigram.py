import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# hyperparameters
batch_size = 32 # independent sequences fed in one pass
block_size = 8 # context length
max_iters = 10_000
eval_interval = 250
lr = 1e-2
eval_iters = 200
n_embd = 32
device = 'cpu' #'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"device: {device}")

torch.manual_seed(1332) # for reproducibility

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
text = open('input.txt', 'r').read()

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

# simple bigram
import torch.nn as nn

class BigramLLM(nn.Module):
    def __init__(self):
        super().__init__()
        # each token reads logits for the next token from lookup table
        self.token_emb_table = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        # idx and targets are (B, T)
        # B -> batch dim
        # T -> time dim -> ctx
        # after embedding them, each idx[i, j] will be have emb dim
        # it'll become (B, T, C) -> C is emb dim
        t_embd = self.token_emb_table(idx)
        logits = self.lm_head(t_embd) # (B, T, vocab_size)
        
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
            # predictions
            logits, loss = self(idx)
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
print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long), 1000)[0].tolist()))
