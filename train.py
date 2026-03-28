
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import pickle
import os
import time


device = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)
print(f"[INFO] Using device: {device}")
if device == 'cuda':
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")



batch_size    = 32        
block_size    = 128       
max_iters     = 8000    
learning_rate = 3e-4
eval_interval = 500
eval_iters    = 100
n_embd        = 256       
n_head        = 8         
n_layer       = 6         
dropout       = 0.15

print(f"""
[CONFIG]
  Embedding dim : {n_embd}
  Attention heads: {n_head}
  Layers        : {n_layer}
  Block size    : {block_size}
  Max iters     : {max_iters}
""")

# ── Load & Prepare Data 
DATA_FILE = "training_data.txt"

if not os.path.exists(DATA_FILE):
    print(f"""
[ERROR] Dataset file not found: '{DATA_FILE}'




""")
    exit(1)

with open(DATA_FILE, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"[INFO] Loaded '{DATA_FILE}': {len(text):,} characters")

# Build vocabulary from characters in the text
chars      = sorted(list(set(text)))
vocab_size = len(chars)
print(f"[INFO] Vocabulary size: {vocab_size} unique characters")

# Encoder / decoder
stoi   = {ch: i for i, ch in enumerate(chars)}
itos   = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

# Train / validation split (90/10)
data        = torch.tensor(encode(text), dtype=torch.long)
n           = int(0.9 * len(data))
train_data  = data[:n]
val_data    = data[n:]
print(f"[INFO] Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

# Save vocab for the chatbot
with open('vocab.pkl', 'wb') as f:
    pickle.dump({'stoi': stoi, 'itos': itos, 'vocab_size': vocab_size}, f)
print("[INFO] Vocabulary saved → vocab.pkl")


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix   = torch.randint(len(data) - block_size, (batch_size,))
    x    = torch.stack([data[i:i + block_size] for i in ix])
    y    = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ── Model Architecture 
class CausalSelfAttention(nn.Module):
    

    def __init__(self):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head  = n_head
        self.head_dim = n_embd // n_head

        # Fused QKV — one linear instead of three (faster, fewer params)
        self.c_attn  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj  = nn.Linear(n_embd, n_embd,     bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv      = self.c_attn(x)
        q, k, v  = qkv.split(C, dim=2)

       
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Flash Attention when available, otherwise standard SDPA
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal  = True,
            dropout_p  = dropout if self.training else 0.0
        )

        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
   

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.SiLU(),                        # Swish — modern default
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm transformer block (GPT-2 / LLaMA style).
    Pre-LN is more stable than Post-LN, especially for deeper models.
    """
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln2  = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual + attention
        x = x + self.ffwd(self.ln2(x))   # residual + FFN
        return x


class CriminalMindGPT(nn.Module):
    """
    Decoder-only GPT language model.
    Trained on criminology text to power a psychology-aware chatbot.
    """
    def __init__(self):
        super().__init__()
        self.token_emb    = nn.Embedding(vocab_size, n_embd)
        self.position_emb = nn.Embedding(block_size, n_embd)
        self.drop         = nn.Dropout(dropout)
        self.blocks       = nn.Sequential(*[TransformerBlock() for _ in range(n_layer)])
        self.ln_f         = nn.LayerNorm(n_embd)
        self.lm_head      = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: input embeddings share weights with output projection
        # Reduces parameters by ~vocab_size × n_embd and improves performance
        self.token_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)

        # GPT-2 style scaled init for residual projections
        for name, p in self.named_parameters():
            if name.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

        total = sum(p.numel() for p in self.parameters())
        print(f"[INFO] Model parameters: {total:,} ({total/1e6:.2f}M)")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= block_size, f"Sequence length {T} exceeds block_size {block_size}"

        tok_emb = self.token_emb(idx)
        pos_emb = self.position_emb(torch.arange(T, device=device))
        x       = self.drop(tok_emb + pos_emb)
        x       = self.blocks(x)
        x       = self.ln_f(x)
        logits  = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss    = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.85, top_k=50, top_p=0.92):
        """
        Nucleus (top-p) + top-k sampling with temperature.
        - temperature : higher = more creative, lower = more focused
        - top_k       : keep only top-k tokens before sampling
        - top_p       : nucleus sampling — keeps smallest set summing to p
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _= self.forward(idx_cond)
            logits   = logits[:, -1, :] / temperature

            # Top-k filter
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filter
            if top_p is not None:
                probs_sorted, sorted_idx = torch.sort(F.softmax(logits, dim=-1), descending=True)
                cum_probs = torch.cumsum(probs_sorted, dim=-1)
                # Remove tokens beyond nucleus
                sorted_remove = (cum_probs - probs_sorted) > top_p
                probs_sorted[sorted_remove] = 0.0
                probs_sorted /= probs_sorted.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs_sorted, num_samples=1)
                idx_next   = sorted_idx.gather(-1, next_token)
            else:
                probs    = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ── Training ──────────────────────────────────────────────────────────────────

model = CriminalMindGPT().to(device)

# torch.compile() requires Triton which is NOT supported on Windows.

print("[INFO] torch.compile() skipped (Windows — Triton not supported)")


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y       = get_batch(split)
            _, loss    = model(X, Y)
            losses[k]  = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



optimizer = torch.optim.AdamW(
    model.parameters(),
    lr           = learning_rate,
    betas        = (0.9, 0.95),
    weight_decay = 0.1
)

# Cosine LR 
def get_lr(step):
    warmup = 200
    min_lr = learning_rate / 10
    if step < warmup:
        return learning_rate * (step + 1) / warmup
    decay = (step - warmup) / (max_iters - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
    return min_lr + coeff * (learning_rate - min_lr)


print("\n[TRAINING STARTED]\n" + "─" * 50)
start_time = time.time()
best_val_loss = float('inf')

for step in range(max_iters):
    # Update learning rate
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    # Evaluate periodically
    if step % eval_interval == 0 or step == max_iters - 1:
        losses  = estimate_loss()
        elapsed = (time.time() - start_time) / 60
        eta     = (elapsed / max(step, 1)) * (max_iters - step)
        print(f"  step {step:4d}/{max_iters} | "
              f"train: {losses['train']:.4f} | val: {losses['val']:.4f} | "
              f"lr: {lr:.2e} | elapsed: {elapsed:.1f}m | eta: {eta:.1f}m")

        
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save({
                'step'      : step,
                'model_state': model.state_dict(),
                'val_loss'  : best_val_loss,
                'config'    : {
                    'n_embd'    : n_embd,
                    'n_head'    : n_head,
                    'n_layer'   : n_layer,
                    'block_size': block_size,
                    'vocab_size': vocab_size,
                }
            }, 'best_model.pt')

    # Forward + backward pass
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # gradient clipping
    optimizer.step()

total_time = (time.time() - start_time) / 60
print(f"\n[TRAINING COMPLETE] Total time: {total_time:.1f} minutes")
print(f"[INFO] Best validation loss: {best_val_loss:.4f}")
print("[INFO] Best model saved → best_model.pt")
print("[INFO] Run chatbot.py to start chatting!")
