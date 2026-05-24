# 📚 Chess Opening Meta-Learning: Project Documentation & Code Breakdown

## 🔍 How the Project Works (High-Level Flow)
This project tackles **next-move prediction in chess openings** under a zero-shot, few-shot meta-learning framework. Instead of memorizing all openings, the model learns *how to quickly adapt* to a completely unseen opening using just a handful of example positions.

1. **Data Preparation**: Loads curated chess opening lines (Lichess dataset), converts them to board states (FEN/EPD), and extracts UCI moves. Each opening becomes a "task".
2. **Context Encoding**: Converts the board into a 64-square token sequence + global features (castling rights, en passant, halfmove clock, side to move).
3. **Model Architecture**: A lightweight Transformer processes the board, predicts a probability distribution over all 8,064 legal UCI moves, and applies a legal-move mask to prevent illegal predictions.
4. **Training Regimes**:
   - **Baseline**: Standard supervised training on all non-held-out openings.
   - **MAML (Meta-Learning)**: Episodic training that simulates adaptation. For each batch, it performs a few inner gradient steps on a small "support" set, then evaluates on a "query" set to update the meta-weights.
5. **Zero-Shot Evaluation**: A completely held-out opening is reserved. The model adapts using 4 support examples, then predicts the next move for 4 unseen query positions. Top-1 accuracy & cross-entropy loss are tracked before/after adaptation.
6. **Current Issue**: Starting adaptation from ply 1–2 (move 1) provides too little positional information, causing unstable gradients. The plan is to shift the evaluation start to the **5th move** (~ply 8–9) where positional structure is clearer.

---

## 📦 Libraries & Dependencies

| Library | Purpose |
|--------|---------|
| `torch`, `torch.nn`, `torch.nn.functional` | Core deep learning framework, model architecture, loss functions, and MAML gradient computation |
| `torch.func.functional_call` (or fallback) | Executes forward pass with adapted parameters without modifying the original model weights |
| `python-chess` | Chess rules engine: validates moves, converts UCI/PGN to board states, generates legal move masks |
| `datasets` (Hugging Face) | Efficiently loads and streams the `Lichess/chess-openings` dataset |
| `pandas`, `numpy` | Data manipulation, batch tensor conversion, numerical operations |
| `matplotlib`, `tqdm` | Training visualization, progress bars, loss/accuracy plotting |
| `dataclasses`, `typing`, `copy`, `random` | Configuration management, type hints, deep copying models for adaptation, reproducibility |

---

## 🧩 Block-by-Block Code Explanation

### **Block 1: Dependency Installation**
```python
!pip -q install datasets python-chess pandas matplotlib tqdm torch