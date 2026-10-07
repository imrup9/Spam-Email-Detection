import argparse
import os
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from model import SimpleTokenizer, SpamLSTMClassifier


class EmailDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def clean_text(text, stop_words):
    if not isinstance(text, str):
        return ""
    text = text.replace("Subject", "")
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = [w.lower() for w in text.split() if w.lower() not in stop_words]
    return " ".join(words)


def train(args):
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Using device: {device}")

    # Ensure stopwords are available
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))

    if not os.path.exists(args.data):
        if os.path.exists(os.path.join("data", args.data)):
            args.data = os.path.join("data", args.data)
        else:
            raise FileNotFoundError(f"Dataset file '{args.data}' not found. Please provide a valid CSV path.")

    print(f"[+] Loading dataset from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"[+] Total raw emails: {len(df)}")

    # Balance classes (downsample majority class 'ham')
    ham_msg = df[df["label"] == "ham"]
    spam_msg = df[df["label"] == "spam"]
    print(f"    - Ham count: {len(ham_msg)}, Spam count: {len(spam_msg)}")

    ham_balanced = ham_msg.sample(n=len(spam_msg), random_state=42)
    balanced_df = pd.concat([ham_balanced, spam_msg]).reset_index(drop=True)
    print(f"[+] Balanced dataset size: {len(balanced_df)} (Spam: {len(spam_msg)}, Ham: {len(ham_balanced)})")

    # Clean text
    print("[+] Preprocessing text...")
    balanced_df["cleaned_text"] = balanced_df["text"].apply(lambda t: clean_text(t, stop_words))
    balanced_df["target"] = (balanced_df["label"] == "spam").astype(int)

    # Train-test split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        balanced_df["cleaned_text"], balanced_df["target"], test_size=0.2, random_state=42, stratify=balanced_df["target"]
    )

    # Tokenizer
    print("[+] Building vocabulary...")
    tokenizer = SimpleTokenizer()
    tokenizer.fit_on_texts(train_texts)
    print(f"[+] Vocabulary size: {tokenizer.vocab_size}")

    train_seqs = tokenizer.pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=args.max_len)
    test_seqs = tokenizer.pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=args.max_len)

    train_dataset = EmailDataset(train_seqs, train_labels)
    test_dataset = EmailDataset(test_seqs, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = SpamLSTMClassifier(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dense_dim=args.dense_dim,
        dropout=args.dropout
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_acc = 0.0
    patience_counter = 0

    print("[+] Starting PyTorch training...")
    for epoch in range(1, args.epochs + 1):
        # Training Phase
        model.train()
        train_loss, train_correct, total_train = 0.0, 0, 0
        for seqs, targets in train_loader:
            seqs, targets = seqs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(targets)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == targets).sum().item()
            total_train += len(targets)

        train_loss /= total_train
        train_acc = train_correct / total_train

        # Validation Phase
        model.eval()
        val_loss, val_correct, total_val = 0.0, 0, 0
        with torch.no_grad():
            for seqs, targets in test_loader:
                seqs, targets = seqs.to(device), targets.to(device)
                logits = model(seqs)
                loss = criterion(logits, targets)

                val_loss += loss.item() * len(targets)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == targets).sum().item()
                total_val += len(targets)

        val_loss /= total_val
        val_acc = val_correct / total_val

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # Early Stopping & Checkpoint Saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if os.path.dirname(args.save_model):
                os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
            if os.path.dirname(args.save_vocab):
                os.makedirs(os.path.dirname(args.save_vocab), exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": tokenizer.vocab_size,
                "max_len": args.max_len,
                "best_val_acc": best_val_acc,
                "val_loss": val_loss
            }, args.save_model)
            tokenizer.save_vocab(args.save_vocab)
            print(f"  --> Checkpoint saved! Best Val Acc: {best_val_acc*100:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[!] Early stopping triggered at epoch {epoch}.")
                break

    print(f"\n[+] Training complete. Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"[+] Model saved to: {args.save_model}")
    print(f"[+] Vocab saved to: {args.save_vocab}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PyTorch LSTM Spam Classifier")
    parser.add_argument("--data", type=str, default="Emails.csv", help="Path to email dataset CSV")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--max_len", type=int, default=100, help="Maximum sequence length")
    parser.add_argument("--embedding_dim", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=16, help="LSTM hidden units")
    parser.add_argument("--dense_dim", type=int, default=32, help="Dense layer hidden units")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--save_model", type=str, default="checkpoints/spam_model.pt", help="Path to save best weights")
    parser.add_argument("--save_vocab", type=str, default="checkpoints/vocab.json", help="Path to save vocabulary")

    args = parser.parse_args()
    train(args)
