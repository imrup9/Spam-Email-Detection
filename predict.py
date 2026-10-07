import argparse
import os
import string
import nltk
from nltk.corpus import stopwords
import torch

from model import SimpleTokenizer, SpamLSTMClassifier


def clean_text(text):
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))

    text = text.replace("Subject", "")
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = [w.lower() for w in text.split() if w.lower() not in stop_words]
    return " ".join(words)


def predict(email_text, model_path=None, vocab_path=None, threshold=0.5):
    # Auto-resolve checkpoint paths
    if model_path is None:
        model_path = "checkpoints/spam_model.pt" if os.path.exists("checkpoints/spam_model.pt") else "spam_model.pt"
    if vocab_path is None:
        vocab_path = "checkpoints/vocab.json" if os.path.exists("checkpoints/vocab.json") else "vocab.json"

    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"Checkpoint '{model_path}' or vocab '{vocab_path}' not found. Please train the model first using train_pytorch.py."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer & checkpoint
    tokenizer = SimpleTokenizer.load_vocab(vocab_path)
    checkpoint = torch.load(model_path, map_location=device)

    model = SpamLSTMClassifier(vocab_size=checkpoint["vocab_size"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Preprocess and prepare sequence
    cleaned = clean_text(email_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    max_len = checkpoint.get("max_len", 100)
    padded_seq = tokenizer.pad_sequences(seq, maxlen=max_len).to(device)

    with torch.no_grad():
        logit = model(padded_seq)
        prob = torch.sigmoid(logit).item()

    is_spam = prob >= threshold
    confidence = prob if is_spam else (1.0 - prob)
    label = "SPAM" if is_spam else "HAM (Legitimate)"

    return label, confidence, prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify raw email text as Spam or Ham")
    parser.add_argument("text", type=str, nargs="?", default="Congratulations! You won a $1,000 Walmart Gift Card. Claim now!", help="Email text to classify")
    parser.add_argument("--model", type=str, default=None, help="Path to saved model checkpoint")
    parser.add_argument("--vocab", type=str, default=None, help="Path to saved vocabulary file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification probability threshold")

    args = parser.parse_args()
    label, conf, prob = predict(args.text, args.model, args.vocab, args.threshold)
    print("\n" + "=" * 50)
    print(f"[+] Input Email : {args.text}")
    print(f"[+] Prediction  : {label}")
    print(f"[+] Confidence  : {conf * 100:.2f}% (Spam Probability: {prob:.4f})")
    print("=" * 50 + "\n")
