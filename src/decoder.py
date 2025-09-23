import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionHead(nn.Module):
    """Single Attention Head with Masking Support.

    This class implements a single attention head which is part of the
    multi-head attention mechanism. It computes the attention for a given
    query, key, and value, with support for an optional causal mask.

    Args:
        d_model (int): The dimension of the input embeddings.
        d_k (int): The dimension of the key vectors.
        d_q (int): The dimension of the query vectors.
        d_v (int): The dimension of the value vectors.

    Attributes:
        wq (nn.Linear): Linear layer to project input to query vectors.
        wk (nn.Linear): Linear layer to project input to key vectors.
        wv (nn.Linear): Linear layer to project input to value vectors.
    """

    def __init__(self, d_model: int, d_k: int, d_q: int, d_v: int):
        super(AttentionHead, self).__init__()

        self.wq = nn.Linear(d_model, d_q, bias=False)
        self.wk = nn.Linear(d_model, d_k, bias=False)
        self.wv = nn.Linear(d_model, d_v, bias=False)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Calculate the attention weights with optional causal mask.

        Args:
            q (Tensor): Query tensor of shape (batch_size, seq_len, d_q).
            k (Tensor): Key tensor of shape (batch_size, seq_len, d_k).
            v (Tensor): Value tensor of shape (batch_size, seq_len, d_v).
            mask (Tensor, optional): Causal mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            Tensor: Output tensor after applying attention.
            Tensor: Attention weights.
        """
        dim_k = k.shape[-1]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dim_k)
        if mask is not None:
            mask_bool = mask.bool()
            scores = scores.masked_fill(~mask_bool,float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)
        return output, weights

    def forward(self, x, mask=None):
        """Forward pass for the attention head with optional causal mask.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor, optional): Causal mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_v).
        """
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        output, _ = self.scaled_dot_product_attention(q, k, v, mask)

        return output

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism with Masking Support.

    This class implements the multi-head attention mechanism, allowing
    the model to focus on different parts of the input sequence at each layer,
    with support for an optional causal mask.

    Args:
        d_model (int): The dimension of the input embeddings.
        num_attention_heads (int): The number of attention heads.

    Attributes:
        heads (nn.ModuleList): A list of attention heads.
        output_linear (nn.Linear): Linear layer to project concatenated heads back to d_model.
    """

    def __init__(self, d_model: int, num_attention_heads: int):
        super(MultiHeadAttention, self).__init__()
        
        dim_head = d_model // num_attention_heads
        self.heads = nn.ModuleList(
            AttentionHead(d_model, dim_head, dim_head, dim_head)
            for _ in range(num_attention_heads)
        )
        self.output_linear = nn.Linear(d_model, d_model, bias=False)


    def forward(self, hidden_state, mask=None):
        """Forward pass for the multi-head attention layer with optional causal mask.

        Args:
            hidden_state (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor, optional): Causal mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        outputs = [h(hidden_state, mask) for h in self.heads]
        concat = torch.cat(outputs, dim=-1)
        x = self.output_linear(concat)
        return x

class FeedForward(nn.Module):
    """FeedForward module for the Transformer.

    This class implements the feed-forward network used in the Transformer
    model. It consists of two linear layers with a GELU activation in between.

    Args:
        d_model (int): The dimension of the input and output embeddings.
        intermediate_size (int): The dimension of the intermediate layer.

    Attributes:
        linear_1 (nn.Linear): The first linear layer that projects from d_model to intermediate_size.
        linear_2 (nn.Linear): The second linear layer that projects from intermediate_size back to d_model.
        gelu (nn.GELU): GELU activation function applied after the first linear layer.
    """

    def __init__(self, d_model: int, intermediate_size: int):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, intermediate_size)
        self.linear_2 = nn.Linear(intermediate_size, d_model)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        l1 = self.linear_1(x)
        g = self.gelu(l1)
        output = self.linear_2(g)
        return output

class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer.

    This class implements a single layer of the Transformer decoder, consisting
    of a masked multi-head self-attention mechanism followed by a feed-forward neural network.
    Both sub-layers are surrounded by residual connections and layer normalization.

    Args:
        d_model (int): The dimension of the input embeddings.
        num_attention_heads (int): The number of attention heads.
        intermediate_size (int): The dimension of the feed-forward network's intermediate layer.

    Attributes:
        layer_norm_1 (nn.LayerNorm): Layer normalization before self-attention.
        layer_norm_2 (nn.LayerNorm): Layer normalization before feed-forward network.
        self_attention (MultiHeadAttention): Masked multi-head self-attention mechanism.
        feed_forward (FeedForward): Feed-forward neural network.
    """

    def __init__(self, d_model: int, num_attention_heads: int, intermediate_size: int):
        super(TransformerDecoderLayer, self).__init__()

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_attention_heads)
        self.feed_forward = FeedForward(d_model, intermediate_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer decoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): Causal mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        hidden_state_att = self.layer_norm_1(x)
        attention = self.attention(hidden_state_att, mask)
        att_x = x + attention
        
        hidden_state_ff = self.layer_norm_2(att_x)
        feed_forward = self.feed_forward(hidden_state_ff)
        out_x = att_x + feed_forward

        return out_x

class Embeddings(nn.Module):
    """Embeddings module for the Transformer.

      This module combines token embeddings and positional embeddings and applies
      layer normalization.

      Args:
          vocab_size (int): The size of the vocabulary.
          max_position_embeddings (int): The maximum number of positions for positional embeddings.
          d_model (int): The dimension of the input embeddings.

      Attributes:
          token_embeddings (nn.Embedding): Embedding layer for token embeddings.
          position_embeddings (nn.Embedding): Embedding layer for positional embeddings.
          layer_norm (nn.LayerNorm): Layer normalization applied after combining embeddings.
      """

    def __init__(
        self,
        vocab_size: int,
        max_position_embeddings: int,
        d_model: int,
        p_dropout: float = 0.0,
    ):
        super(Embeddings, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass to combine token and positional embeddings.

        Args:
            input_ids (torch.Tensor): Tensor containing input token IDs of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The combined and normalized embeddings of shape (batch_size, seq_len, d_model).
        """
        # Generate position IDs based on the input sequence length
        batch_size, seq_length = input_ids.size()
        position_ids = (
            torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_length)
        )

        # Create token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # Combine token and position embeddings
        sum_emb = token_embeddings + position_embeddings
        ln = self.layer_norm(sum_emb)
        embeddings = self.dropout(ln)

        return embeddings

class TransformerDecoder(nn.Module):
    """Transformer Decoder.

    This class implements the decoder part of the Transformer model, consisting
    of an embeddings layer followed by a stack of Transformer decoder layers.

    Args:
        vocab_size (int): The size of the vocabulary.
        max_position_embeddings (int): The maximum number of positions for positional embeddings.
        d_model (int): The dimension of the input embeddings.
        num_attention_heads (int): The number of attention heads.
        intermediate_size (int): The dimension of the feed-forward network's intermediate layer.
        num_hidden_layers (int): The number of Transformer decoder layers to stack.

    Attributes:
        embeddings (Embeddings): Embeddings layer combining token and positional embeddings.
        layers (nn.ModuleList): List of Transformer decoder layers.
    """

    def __init__(self, vocab_size: int, max_position_embeddings: int, d_model: int,
                 num_attention_heads: int, intermediate_size: int, num_hidden_layers: int):
        super(TransformerDecoder, self).__init__()
        self.embeddings = Embeddings(vocab_size, max_position_embeddings, d_model)
        self.layers = nn.ModuleList(
            TransformerDecoderLayer(d_model, num_attention_heads, intermediate_size)
            for _ in range(num_hidden_layers)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer decoder.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        seq_len = input_ids.size(1)
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=0
        ) 
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class TransformerForLanguageModeling(nn.Module):
    """Transformer model with a language modeling head for text generation.

    Args:
        vocab_size (int): Vocabulary size.
        max_position_embeddings (int): Maximum number of position embeddings.
        d_model (int): Hidden size of the Transformer model.
        num_attention_heads (int): Number of attention heads in each decoder layer.
        intermediate_size (int): Intermediate size of the feed-forward network.
        num_hidden_layers (int): Number of Transformer decoder layers.

    Attributes:
        transformer_decoder (TransformerDecoder): The Transformer decoder.
        lm_head (nn.Linear): Linear layer mapping hidden states to vocabulary logits.
    """

    def __init__(self, vocab_size: int, max_position_embeddings: int, d_model: int,
                 num_attention_heads: int, intermediate_size: int, num_hidden_layers: int):
        super(TransformerForLanguageModeling, self).__init__()
        self.transformer_decoder = TransformerDecoder(
            vocab_size,
            max_position_embeddings,
            d_model,
            num_attention_heads,
            intermediate_size,
            num_hidden_layers
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer model with language modeling head.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        hidden_states = self.transformer_decoder(input_ids)
        logits = self.lm_head(hidden_states)
        return logits
