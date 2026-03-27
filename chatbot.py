"""
Criminal Mind GPT — Interactive Chatbot
=========================================
Back-and-forth conversation interface for the trained GPT model.

Usage:
    python chatbot.py

Commands inside chat:
    /clear      — reset conversation history
    /temp 0.9   — change temperature on the fly
    /topk 40    — change top-k value
    /topp 0.92  — change top-p value
    /tokens 150 — change max response length
    /help       — show all commands
    /quit       — exit
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import math
import os
import sys

# ── Device ────────────────────────────────────────────────────────────────────
device = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

# ── Load Vocabulary ───────────────────────────────────────────────────────────
VOCAB_FILE = 'vocab.pkl'
MODEL_FILE = 'best_model.pt'

if not os.path.exists(VOCAB_FILE):
    print(f"[ERROR] '{VOCAB_FILE}' not found. Run train.py first.")
    sys.exit(1)

if not os.path.exists(MODEL_FILE):
    print(f"[ERROR] '{MODEL_FILE}' not found. Run train.py first.")
    sys.exit(1)

with open(VOCAB_FILE, 'rb') as f:
    vocab = pickle.load(f)

stoi       = vocab['stoi']
itos       = vocab['itos']
vocab_size = vocab['vocab_size']

encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

# ── Load Model Config from Checkpoint ────────────────────────────────────────
checkpoint = torch.load(MODEL_FILE, map_location=device)
cfg        = checkpoint['config']

n_embd     = cfg['n_embd']
n_head     = cfg['n_head']
n_layer    = cfg['n_layer']
block_size = cfg['block_size']
dropout    = 0.0   # disable dropout at inference

# ── Model Definition (must match train.py) ───────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_head   = n_head
        self.head_dim = n_embd // n_head
        self.c_attn   = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj   = nn.Linear(n_embd, n_embd,     bias=False)
        self.drop     = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.c_proj(out))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.SiLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln2  = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class CriminalMindGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb    = nn.Embedding(vocab_size, n_embd)
        self.position_emb = nn.Embedding(block_size, n_embd)
        self.drop         = nn.Dropout(dropout)
        self.blocks       = nn.Sequential(*[TransformerBlock() for _ in range(n_layer)])
        self.ln_f         = nn.LayerNorm(n_embd)
        self.lm_head      = nn.Linear(n_embd, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight

    def forward(self, idx):
        B, T    = idx.shape
        tok_emb = self.token_emb(idx)
        pos_emb = self.position_emb(torch.arange(T, device=device))
        x       = self.drop(tok_emb + pos_emb)
        x       = self.blocks(x)
        return self.lm_head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.85, top_k=50, top_p=0.92):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits   = self.forward(idx_cond)[:, -1, :] / temperature

            # Top-k
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus)
            probs_sorted, sorted_idx = torch.sort(F.softmax(logits, dim=-1), descending=True)
            cum = torch.cumsum(probs_sorted, dim=-1)
            probs_sorted[(cum - probs_sorted) > top_p] = 0.0
            probs_sorted /= probs_sorted.sum(dim=-1, keepdim=True)
            next_tok = torch.multinomial(probs_sorted, 1)
            idx      = torch.cat((idx, sorted_idx.gather(-1, next_tok)), dim=1)
        return idx


# ── Load Weights ──────────────────────────────────────────────────────────────
model = CriminalMindGPT().to(device)

# Strip 'torch.compile' prefix if present
state_dict = checkpoint['model_state']
state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

# ── Conversation Engine ───────────────────────────────────────────────────────

class Conversation:
    """
    Manages multi-turn conversation context.
    Keeps a rolling window of the last N characters to stay within block_size.
    """
    def __init__(self, max_context_chars=400):
        self.history          = []          # list of (role, text) tuples
        self.max_context_chars = max_context_chars

    def add(self, role, text):
        self.history.append((role, text))

    def get_prompt(self, user_input):
        """Build a prompt string from conversation history + new input."""
        self.add("Human", user_input)
        lines = []
        for role, text in self.history:
            lines.append(f"{role}: {text}")
        lines.append("Bot:")
        full = "\n".join(lines)
        # Keep within context window (last max_context_chars chars)
        return full[-self.max_context_chars:]

    def add_response(self, text):
        self.add("Bot", text)

    def clear(self):
        self.history = []
        print("\n[Conversation cleared]\n")


def extract_response(generated: str, prompt: str) -> str:
    """
    Extract only the bot's reply from the generated text.
    Stops at the next 'Human:' turn or newline after first sentence.
    """
    # Get only the new part after the prompt
    new_text = generated[len(prompt):]

    # Stop at next Human turn
    if "Human:" in new_text:
        new_text = new_text[:new_text.index("Human:")]

    # Clean up whitespace
    new_text = new_text.strip()

    # Fallback: return at least something
    if not new_text:
        new_text = "..."

    return new_text


def chat(user_input: str, conv: Conversation, temperature=0.85,
         top_k=50, top_p=0.92, max_tokens=150) -> str:
    """Generate a response for a user input."""
    prompt  = conv.get_prompt(user_input)
    encoded = encode(prompt)

    if len(encoded) == 0:
        conv.add_response("I couldn't process that.")
        return "I couldn't process that."

    context = torch.tensor([encoded], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model.generate(
            context,
            max_new_tokens = max_tokens,
            temperature    = temperature,
            top_k          = top_k,
            top_p          = top_p
        )

    generated  = decode(output[0].tolist())
    response   = extract_response(generated, prompt)
    conv.add_response(response)
    return response


# ── CLI Interface ─────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════╗
║          CRIMINAL MIND GPT  —  Interactive Chat          ║
║    Trained on "Inside the Criminal Mind" — Samenow       ║
╚══════════════════════════════════════════════════════════╝

  Type your message and press Enter to chat.
  Type /help for commands.
"""

HELP_TEXT = """
Commands:
  /clear         — clear conversation history
  /temp <float>  — set temperature  (default: 0.85)
  /topk <int>    — set top-k        (default: 50)
  /topp <float>  — set top-p        (default: 0.92)
  /tokens <int>  — set max tokens   (default: 150)
  /history       — show conversation so far
  /help          — show this message
  /quit          — exit
"""

def run_chatbot():
    print(BANNER)
    print(f"  Device  : {device}")
    print(f"  Model   : {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Context : {block_size} tokens\n")

    conv        = Conversation()
    temperature = 0.85
    top_k       = 50
    top_p       = 0.92
    max_tokens  = 150

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        # Commands
        if user_input.startswith('/'):
            parts = user_input.split()
            cmd   = parts[0].lower()

            if cmd == '/quit':
                print("Goodbye!")
                break
            elif cmd == '/clear':
                conv.clear()
            elif cmd == '/help':
                print(HELP_TEXT)
            elif cmd == '/history':
                print("\n── Conversation History ──")
                for role, text in conv.history:
                    print(f"  {role}: {text}")
                print("─────────────────────────\n")
            elif cmd == '/temp' and len(parts) == 2:
                temperature = float(parts[1])
                print(f"  [temperature set to {temperature}]")
            elif cmd == '/topk' and len(parts) == 2:
                top_k = int(parts[1])
                print(f"  [top-k set to {top_k}]")
            elif cmd == '/topp' and len(parts) == 2:
                top_p = float(parts[1])
                print(f"  [top-p set to {top_p}]")
            elif cmd == '/tokens' and len(parts) == 2:
                max_tokens = int(parts[1])
                print(f"  [max tokens set to {max_tokens}]")
            else:
                print(f"  [Unknown command. Type /help]")
            continue

        # Generate response
        print("Bot: ", end='', flush=True)
        response = chat(
            user_input,
            conv,
            temperature = temperature,
            top_k       = top_k,
            top_p       = top_p,
            max_tokens  = max_tokens
        )
        print(response)
        print()


if __name__ == '__main__':
    run_chatbot()
