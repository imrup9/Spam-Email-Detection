import json
import re
import string
import torch
import torch.nn as nn


class SimpleTokenizer:
    """
    Production-grade text tokenizer and vocabulary manager for PyTorch.
    Handles tokenization, vocab indexing, special tokens (<PAD>, <UNK>),
    and sequence padding/truncation.
    """
    def __init__(self, pad_token="<PAD>", unk_token="<UNK>", max_vocab_size=None):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.max_vocab_size = max_vocab_size
        self.word2idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx2word = {0: self.pad_token, 1: self.unk_token}

    def fit_on_texts(self, texts):
        word_freq = {}
        for text in texts:
            tokens = text.lower().split()
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        if self.max_vocab_size:
            sorted_words = sorted_words[: self.max_vocab_size - 2]

        for word, _ in sorted_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def texts_to_sequences(self, texts):
        sequences = []
        unk_idx = self.word2idx[self.unk_token]
        for text in texts:
            tokens = text.lower().split()
            seq = [self.word2idx.get(token, unk_idx) for token in tokens]
            sequences.append(seq)
        return sequences

    def pad_sequences(self, sequences, maxlen=100, padding='pre'):
        padded = []
        pad_idx = self.word2idx[self.pad_token]
        for seq in sequences:
            if len(seq) >= maxlen:
                padded_seq = seq[-maxlen:] if padding == 'pre' else seq[:maxlen]
            else:
                if padding == 'pre':
                    padded_seq = [pad_idx] * (maxlen - len(seq)) + seq
                else:
                    padded_seq = seq + [pad_idx] * (maxlen - len(seq))
            padded.append(padded_seq)
        return torch.tensor(padded, dtype=torch.long)

    def save_vocab(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"word2idx": self.word2idx, "maxlen": 100}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_vocab(cls, filepath):
        tokenizer = cls()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer.word2idx = data["word2idx"]
        tokenizer.idx2word = {int(v): k for k, v in tokenizer.word2idx.items()}
        return tokenizer

    @property
    def vocab_size(self):
        return len(self.word2idx)


class SpamLSTMClassifier(nn.Module):
    """
    PyTorch Recurrent Neural Network (LSTM) for binary spam classification.
    """
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=16, dense_dim=32, dropout=0.2):
        super(SpamLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden_dim, dense_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # shape: (batch_size, seq_len, embedding_dim)
        lstm_out, (hn, cn) = self.lstm(embedded)  # hn shape: (1, batch_size, hidden_dim)
        last_hidden = hn[-1]  # shape: (batch_size, hidden_dim)
        
        dense_out = self.fc1(last_hidden)  # shape: (batch_size, dense_dim)
        dense_out = self.relu(dense_out)
        dense_out = self.dropout(dense_out)
        logits = self.fc2(dense_out)  # shape: (batch_size, 1)
        return logits.squeeze(-1)
