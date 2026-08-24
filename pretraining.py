"""
Pretraining de GPT (sección 5.4 del libro "Build a Large Language Model
From Scratch"): entrena GPT-2 small (124M) desde cero.

Dos datasets disponibles:
  - "verdict":  "The Verdict" de Edith Wharton (texto corto, sin dependencias)
  - "fineweb":  codelion/fineweb-edu-1B de HuggingFace (~970k documentos,
                ~1B tokens, en streaming). Requiere: pip install datasets

Solo lo necesario: dataset + dataloader, función de pérdida, bucle de
entrenamiento y generación de muestra. No requiere TensorFlow.

Uso desde terminal:
    python pretraining.py                                  # 10 épocas con The Verdict
    python pretraining.py --dataset fineweb                # FineWeb-Edu en streaming
    python pretraining.py --dataset fineweb --max_docs 2000 --epochs 1
    python pretraining.py --epochs 5 --device cpu          # forzar CPU

Uso desde notebook: importar las funciones (ver celdas en la respuesta).
"""

import urllib.request

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

import tiktoken

from Transformer_arquitectures import GPTModel

# ---------------------------------------------------------------------------
# Configuración del modelo (igual que en el capítulo 5, sección 5.4)
# ---------------------------------------------------------------------------
GPT_CONFIG_124M = {
    "vocab_size": 50257,      # vocabulario de GPT-2
    "context_length": 256,    # ventana de contexto del capítulo 5
    "emb_dim": 768,           # dimensión de embedding
    "n_heads": 12,            # cabezas de atención
    "n_layers": 12,           # bloques transformer
    "drop_rate": 0.0,         # sin dropout durante el pretraining
    "qkv_bias": False         # sin bias en Q/K/V (config del libro)
}

# ---------------------------------------------------------------------------
# Dataset y DataLoader (capítulo 2)
# ---------------------------------------------------------------------------
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def custom_collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    return inputs, targets


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader


# ---------------------------------------------------------------------------
# Helpers de texto
# ---------------------------------------------------------------------------
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # shape: (1, n_tokens)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # Generación greedy (argmax) usada para mostrar muestras durante el training
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# ---------------------------------------------------------------------------
# Pérdida, evaluación y entrenamiento (sección 5.4)
# ---------------------------------------------------------------------------
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(
        logits.flatten(0, 1),  # (batch*seq, vocab)
        target_batch.flatten()  # (batch*seq,)
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    try:
        n_total = len(data_loader)
    except TypeError:
        n_total = None  # dataloader streaming (IterableDataset): no tiene len()

    if n_total == 0:
        return float("nan")
    if num_batches is None:
        if n_total is None:
            raise ValueError(
                "num_batches es obligatorio para dataloaders streaming")
        num_batches = n_total
    elif n_total is not None:
        num_batches = min(num_batches, n_total)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer,
                       eval_train_loader=None, eval_val_loader=None):
    """Bucle de entrenamiento. Por defecto evalúa sobre train_loader/val_loader;
    con datasets streaming pasa loaders fijos en eval_train_loader/eval_val_loader."""
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    eval_train = eval_train_loader or train_loader
    eval_val = eval_val_loader or val_loader

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, eval_train, eval_val, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


# ---------------------------------------------------------------------------
# Dataset de entrenamiento
# ---------------------------------------------------------------------------
VERDICT_URL = ("https://raw.githubusercontent.com/rasbt/"
               "LLMs-from-scratch/main/ch05/"
               "01_main-chapter-code/the-verdict.txt")


def download_verdict(file_path="the-verdict.txt"):
    """Descarga 'The Verdict' (o usa la copia local si ya existe) y la devuelve."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Descargando {file_path} ...")
        urllib.request.urlretrieve(VERDICT_URL, file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


# ---------------------------------------------------------------------------
# FineWeb-Edu (codelion/fineweb-edu-1B): dataset grande de HuggingFace, en
# streaming (no se descarga entero). Mismo patrón que el dataloader del libro
# (ch05/10_llm-training-speed): ventanas sin solapamiento + token <|endoftext|>
# entre documentos. Requiere: pip install datasets
# ---------------------------------------------------------------------------
class TokenBlockIterableDataset(IterableDataset):
    """Streaming: tokeniza documentos de FineWeb-Edu y genera ventanas de
    max_length tokens. Cada documento termina con <|endoftext|>. Los
    documentos se reparten en train/val según val_mod (cada val_mod-ésimo
    documento va a validación)."""

    def __init__(self, stream, encoder, max_length=1024, add_eot=True,
                 val_mod=100, val_split=False, max_docs=None):
        self.stream = stream
        self.max_length = max_length
        self.add_eot = add_eot
        self.enc = encoder
        self.val_mod = val_mod
        self.val_split = val_split
        self.max_docs = max_docs
        self.EOT = encoder.encode(
            "<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    def __iter__(self):
        it = self.stream
        wi = get_worker_info()
        if wi is not None and wi.num_workers > 1:
            it = self.stream.shard(
                num_shards=wi.num_workers, index=wi.id, contiguous=True)

        buf = []
        seen, kept = 0, 0
        for ex in it:
            text = ex.get("text", "") if ex else ""
            if not isinstance(text, str) or not text.strip():
                continue
            if (seen % self.val_mod == 0) != self.val_split:
                seen += 1
                continue
            seen += 1

            toks = self.enc.encode(text)
            if self.add_eot:
                toks.append(self.EOT)
            buf.extend(toks)

            while len(buf) >= self.max_length + 1:
                x = torch.tensor(buf[:self.max_length], dtype=torch.long)
                y = torch.tensor(buf[1:self.max_length + 1], dtype=torch.long)
                yield x, y
                del buf[:self.max_length]

            kept += 1
            if self.max_docs is not None and kept >= self.max_docs:
                break


class FixedPairsDataset(Dataset):
    """Dataset en memoria (con len()) con pares (x, y) ya tokenizados."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def materialize_from_loader(loader, max_batches=32):
    """Congela los primeros max_batches de un loader streaming en un Dataset."""
    pairs = []
    for i, (x, y) in enumerate(loader):
        pairs.extend(list(zip(x, y)))
        if (i + 1) >= max_batches:
            break
    return FixedPairsDataset(pairs)


def make_fixed_eval_loaders(train_loader, val_loader,
                            max_train_batches=8, max_val_batches=16):
    """Loaders fijos (con len()) para evaluar sin perturbar el stream."""
    train_bs = getattr(train_loader, "batch_size", 1) or 1
    val_bs = getattr(val_loader, "batch_size", 1) or 1

    fixed_train_eval = materialize_from_loader(
        train_loader, max_batches=max_train_batches)
    fixed_val_eval = materialize_from_loader(
        val_loader, max_batches=max_val_batches)

    train_eval_loader = DataLoader(
        fixed_train_eval, batch_size=train_bs, shuffle=False, drop_last=True)
    val_eval_loader = DataLoader(
        fixed_val_eval, batch_size=val_bs, shuffle=False, drop_last=True)
    return train_eval_loader, val_eval_loader


def create_dataloader_fineweb(batch_size=2, max_length=1024, val_mod=100,
                              seed=123, max_docs=None, num_workers=0,
                              add_eot=True, shuffle_buffer=10000):
    """Loaders streaming de codelion/fineweb-edu-1B (~970k docs, ~1B tokens).

    Devuelve (train_loader, val_loader). Para evaluar durante el training usa
    make_fixed_eval_loaders(train_loader, val_loader).
    max_docs limita cuántos documentos consume el train (None = todo el
    dataset, ~1B tokens). Cada documento ≈ 1-2k tokens.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Para usar FineWeb-Edu necesitas instalar 'datasets': "
            "pip install datasets") from e

    stream = load_dataset("codelion/fineweb-edu-1B",
                          split="train", streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)

    enc = tiktoken.get_encoding("gpt2")
    train_ds = TokenBlockIterableDataset(
        stream, enc, max_length=max_length, add_eot=add_eot,
        val_mod=val_mod, val_split=False, max_docs=max_docs)
    val_ds = TokenBlockIterableDataset(
        stream, enc, max_length=max_length, add_eot=add_eot,
        val_mod=val_mod, val_split=True, max_docs=None)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False,
        drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        drop_last=True, num_workers=num_workers)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Punto de entrada (también usable como script de terminal)
# ---------------------------------------------------------------------------
def main(epochs=None, batch_size=2, max_length=None, eval_freq=None, eval_iter=1,
         start_context="Every effort moves you", device=None, seed=123,
         dataset="verdict", max_docs=5000, val_mod=100):
    torch.manual_seed(seed)
    device = device or pick_device()
    print(f"Device: {device}")

    if dataset == "fineweb":
        max_length = max_length or 1024
        epochs = epochs or 1
        eval_freq = eval_freq or 100

        train_loader, val_loader = create_dataloader_fineweb(
            batch_size=batch_size, max_length=max_length, val_mod=val_mod,
            seed=seed, max_docs=max_docs)
        train_eval_loader, val_eval_loader = make_fixed_eval_loaders(
            train_loader, val_loader, max_train_batches=8, max_val_batches=16)

        cfg = {**GPT_CONFIG_124M, "context_length": max_length}
        print(f"Dataset: codelion/fineweb-edu-1B (streaming, "
              f"max_docs={max_docs}, 1 doc de val cada {val_mod})")
    else:
        max_length = max_length or 256
        epochs = epochs or 10
        eval_freq = eval_freq or 5

        text_data = download_verdict()
        train_ratio = 0.90
        split_idx = int(train_ratio * len(text_data))

        train_loader = create_dataloader_v1(
            text_data[:split_idx],
            batch_size=batch_size, max_length=max_length,
            stride=max_length // 2, drop_last=True, shuffle=True)
        val_loader = create_dataloader_v1(
            text_data[split_idx:],
            batch_size=batch_size, max_length=max_length,
            stride=max_length // 2, drop_last=False, shuffle=False)

        train_eval_loader = val_eval_loader = None
        cfg = GPT_CONFIG_124M
        print("Dataset: The Verdict")
        print("Training batches:", len(train_loader))
        print("Validation batches:", len(val_loader))

    model = GPTModel(cfg)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    tokenizer = tiktoken.get_encoding("gpt2")

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=epochs, eval_freq=eval_freq, eval_iter=eval_iter,
        start_context=start_context, tokenizer=tokenizer,
        eval_train_loader=train_eval_loader, eval_val_loader=val_eval_loader)

    print(f"\nLoss final de entrenamiento: {train_losses[-1]:.3f}")
    print(f"Loss final de validación:    {val_losses[-1]:.3f}")

    torch.save(model.state_dict(), "gpt2-pretrained.pth")
    print("Modelo guardado en gpt2-pretrained.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pretraining de GPT-2 small (124M)")
    parser.add_argument("--dataset", type=str, default="verdict",
                        choices=["verdict", "fineweb"],
                        help="verdict (The Verdict) o fineweb (FineWeb-Edu)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="por defecto: 10 (verdict) o 1 (fineweb)")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=None,
                        help="por defecto: 256 (verdict) o 1024 (fineweb)")
    parser.add_argument("--eval_freq", type=int, default=None,
                        help="por defecto: 5 (verdict) o 100 (fineweb)")
    parser.add_argument("--eval_iter", type=int, default=1)
    parser.add_argument("--start_context", type=str, default="Every effort moves you")
    parser.add_argument("--device", type=str, default=None,
                        help="cpu, mps o cuda (por defecto: auto)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_docs", type=int, default=5000,
                        help="fineweb: máx. documentos de entrenamiento "
                             "(None = todo el dataset)")
    parser.add_argument("--val_mod", type=int, default=100,
                        help="fineweb: 1 documento de validación cada N")
    args = parser.parse_args()

    main(epochs=args.epochs, batch_size=args.batch_size,
         max_length=args.max_length, eval_freq=args.eval_freq,
         eval_iter=args.eval_iter, start_context=args.start_context,
         device=torch.device(args.device) if args.device else None,
         seed=args.seed, dataset=args.dataset,
         max_docs=args.max_docs, val_mod=args.val_mod)
