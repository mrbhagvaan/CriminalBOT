# Criminal Mind GPT 🧠

A **decoder-only GPT language model** trained on *Inside the Criminal Mind* by Stanton E. Samenow — implemented from scratch in PyTorch with a back-and-forth interactive chatbot interface.

---

## Project Highlights

| Feature | Detail |
|---|---|
| Architecture | Decoder-only Transformer (GPT-style) |
| Training | Character-level language modeling |
| Sampling | Temperature + Top-k + Top-p (nucleus) |
| Interface | Multi-turn CLI chatbot with conversation memory |
| Hardware target | NVIDIA GTX 1650Ti (4GB VRAM) |
| Framework | PyTorch 2.0+ |

---

## Model Architecture

```
CriminalMindGPT
├── Token Embedding       (vocab_size × 256)
├── Position Embedding    (128 × 256)
├── 6 × TransformerBlock
│   ├── LayerNorm (Pre-LN)
│   ├── CausalSelfAttention  — 8 heads, fused QKV, Flash Attention
│   ├── LayerNorm
│   └── FeedForward          — 4× expansion, SiLU activation
├── Final LayerNorm
└── LM Head (weight-tied with token embedding)
```

**Key design choices:**
- **Pre-LayerNorm** — more stable training than Post-LN (GPT-2 / LLaMA style)
- **Fused QKV projection** — one linear instead of three; fewer parameters
- **Weight tying** — input embedding and output projection share weights
- **SiLU activation** — modern alternative to GELU, used in LLaMA/Mistral
- **Flash Attention** — via `scaled_dot_product_attention`, 2–4× faster on CUDA
- **Nucleus sampling** — top-p + top-k for high-quality diverse output

---

## Setup

```bash
# 1. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Add your dataset
# Place the plain-text book as: inside_criminal_mind.txt

# 3. Train the model (~1–2 hours on GTX 1650Ti)
python train.py

# 4. Start chatting
python chatbot.py
```

---

## Chatbot Commands

| Command | Description |
|---|---|
| `/clear` | Reset conversation history |
| `/temp 0.9` | Adjust temperature (creativity) |
| `/topk 40` | Adjust top-k sampling |
| `/topp 0.95` | Adjust nucleus sampling threshold |
| `/tokens 200` | Change max response length |
| `/history` | View conversation so far |
| `/quit` | Exit |

---

## Files

```
.
├── train.py                  # Model definition + training loop
├── chatbot.py                # Interactive chatbot interface
├── inside_criminal_mind.txt  # Dataset (provide your own)
├── best_model.pt             # Saved after training (auto-generated)
└── vocab.pkl                 # Character vocabulary (auto-generated)
```

---

## Tech Stack

- **PyTorch 2.0** — model, training, Flash Attention
- **Pure Python** — no HuggingFace, no external ML libs
- Character-level tokenization (no external tokenizer needed)

---

## Skills Demonstrated

- Transformer architecture implementation from scratch
- GPU-aware training pipeline (CUDA / MPS / CPU fallback)
- Cosine LR scheduling with linear warmup
- Gradient clipping, weight decay, AdamW optimizer
- Nucleus sampling (top-p + top-k) for text generation
- Stateful multi-turn conversation management
- Model checkpointing (saves best val loss)
