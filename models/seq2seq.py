"""
Seq2Seq with Attention Model (Model 2)
=======================================
Bidirectional LSTM Encoder + LSTM Decoder with Bahdanau Attention.
This serves as the baseline/comparison deep learning model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Tuple, Optional


class Encoder(nn.Module):
    """
    Bidirectional LSTM Encoder.
    Encodes the source code tokens into hidden representations.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 300,
                 hidden_dim: int = 512, n_layers: int = 2,
                 dropout: float = 0.3, pad_idx: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """
        Args:
            src: (batch, src_len) source token indices.

        Returns:
            outputs: (batch, src_len, hidden*2) encoder outputs.
            (hidden, cell): Decoder-compatible hidden states.
        """
        # src: (batch, src_len)
        embedded = self.dropout(self.embedding(src))  # (batch, src_len, embed_dim)

        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: (batch, src_len, hidden*2)
        # hidden: (n_layers*2, batch, hidden)

        # Concatenate forward and backward hidden states for decoder
        # hidden: (n_layers, 2, batch, hidden) -> (n_layers, batch, hidden*2)
        hidden = hidden.view(self.n_layers, 2, -1, self.hidden_dim)
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
        hidden = torch.tanh(self.fc_hidden(hidden))

        cell = cell.view(self.n_layers, 2, -1, self.hidden_dim)
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)
        cell = torch.tanh(self.fc_cell(cell))

        return outputs, (hidden, cell)


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism.
    Computes attention weights over encoder outputs given decoder state.
    """

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int = 256):
        super().__init__()
        self.W_encoder = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.W_decoder = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.V = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, decoder_hidden: torch.Tensor,
                encoder_outputs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_hidden: (batch, decoder_dim) current decoder hidden state.
            encoder_outputs: (batch, src_len, encoder_dim) all encoder outputs.
            mask: (batch, src_len) padding mask.

        Returns:
            context: (batch, encoder_dim) weighted context vector.
            attn_weights: (batch, src_len) attention distribution.
        """
        # decoder_hidden: (batch, decoder_dim) -> (batch, 1, decoder_dim)
        decoder_hidden = decoder_hidden.unsqueeze(1)

        # Energy: (batch, src_len, attention_dim)
        energy = torch.tanh(
            self.W_encoder(encoder_outputs) + self.W_decoder(decoder_hidden)
        )

        # Attention scores: (batch, src_len)
        scores = self.V(energy).squeeze(2)

        # Apply mask (set padding positions to -inf)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Attention weights: (batch, src_len)
        attn_weights = F.softmax(scores, dim=1)

        # Context vector: (batch, encoder_dim)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attn_weights


class Decoder(nn.Module):
    """
    LSTM Decoder with Bahdanau Attention.
    Generates docstring tokens one at a time.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 300,
                 hidden_dim: int = 512, encoder_dim: int = 1024,
                 n_layers: int = 2, dropout: float = 0.3, pad_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.attention = BahdanauAttention(encoder_dim, hidden_dim)
        self.rnn = nn.LSTM(
            embed_dim + encoder_dim, hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim + encoder_dim + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt_token: torch.Tensor,
                hidden: Tuple[torch.Tensor, torch.Tensor],
                encoder_outputs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple:
        """
        Single decoding step.

        Args:
            tgt_token: (batch,) current target token.
            hidden: (hidden_state, cell_state) from previous step.
            encoder_outputs: (batch, src_len, encoder_dim).
            mask: Source padding mask.

        Returns:
            prediction: (batch, vocab_size) logits.
            hidden: Updated hidden state.
            attn_weights: Attention weights for visualization.
        """
        tgt_token = tgt_token.unsqueeze(1)  # (batch, 1)
        embedded = self.dropout(self.embedding(tgt_token))  # (batch, 1, embed_dim)

        # Attention
        h_top = hidden[0][-1]  # Top layer hidden: (batch, hidden_dim)
        context, attn_weights = self.attention(h_top, encoder_outputs, mask)
        context = context.unsqueeze(1)  # (batch, 1, encoder_dim)

        # Concatenate embedding + context as RNN input
        rnn_input = torch.cat([embedded, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        # Prediction: combine output, context, embedding
        prediction = self.fc_out(
            torch.cat([output.squeeze(1), context.squeeze(1), embedded.squeeze(1)], dim=1)
        )

        return prediction, hidden, attn_weights


class Seq2Seq(nn.Module):
    """
    Full Seq2Seq model combining Encoder, Decoder, and Attention.

    Model 2 in our project — serves as the baseline deep learning model
    for comparison against CodeBERT.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def create_mask(self, src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """Create padding mask for attention."""
        return (src != pad_idx)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Forward pass with optional teacher forcing.

        Args:
            src: (batch, src_len) source code tokens.
            tgt: (batch, tgt_len) target docstring tokens.
            teacher_forcing_ratio: Probability of using ground truth as next input.

        Returns:
            outputs: (batch, tgt_len, vocab_size) predictions.
        """
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)
        mask = self.create_mask(src)

        # Encode
        encoder_outputs, hidden = self.encoder(src)

        # First decoder input is <SOS> token
        current_token = tgt[:, 0]

        for t in range(1, tgt_len):
            prediction, hidden, _ = self.decoder(current_token, hidden, encoder_outputs, mask)
            outputs[:, t, :] = prediction

            # Teacher forcing decision
            if random.random() < teacher_forcing_ratio:
                current_token = tgt[:, t]
            else:
                current_token = prediction.argmax(dim=1)

        return outputs

    def generate(self, src: torch.Tensor, sos_idx: int, eos_idx: int,
                 max_len: int = 128) -> Tuple[list, list]:
        """
        Generate a docstring for given source code (inference mode).

        Args:
            src: (1, src_len) single source sequence.
            sos_idx: Start of sequence token index.
            eos_idx: End of sequence token index.
            max_len: Maximum generation length.

        Returns:
            tokens: List of generated token indices.
            attention_weights: List of attention weight tensors.
        """
        self.eval()
        with torch.no_grad():
            mask = self.create_mask(src)
            encoder_outputs, hidden = self.encoder(src)
            current_token = torch.tensor([sos_idx]).to(self.device)

            generated_tokens = []
            attention_weights = []

            for _ in range(max_len):
                prediction, hidden, attn = self.decoder(
                    current_token, hidden, encoder_outputs, mask
                )
                token_id = prediction.argmax(dim=1).item()
                generated_tokens.append(token_id)
                attention_weights.append(attn.cpu())

                if token_id == eos_idx:
                    break

                current_token = torch.tensor([token_id]).to(self.device)

        return generated_tokens, attention_weights


def build_seq2seq_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    device: torch.device,
    embed_dim: int = 300,
    hidden_dim: int = 512,
    n_layers: int = 2,
    dropout: float = 0.3
) -> Seq2Seq:
    """
    Factory function to build the complete Seq2Seq model.

    Args:
        src_vocab_size: Source vocabulary size.
        tgt_vocab_size: Target vocabulary size.
        device: torch device (cpu/cuda).
        embed_dim: Embedding dimension.
        hidden_dim: LSTM hidden dimension.
        n_layers: Number of LSTM layers.
        dropout: Dropout probability.

    Returns:
        Initialized Seq2Seq model.
    """
    encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, n_layers, dropout)
    decoder = Decoder(tgt_vocab_size, embed_dim, hidden_dim, hidden_dim * 2, n_layers, dropout)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    model.apply(init_weights)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Seq2Seq Model: {total_params:,} total params, {trainable:,} trainable")

    return model
