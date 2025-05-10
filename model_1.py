## Building and training a bigram language model
from functools import partial
import math
import torch
import torch.nn as nn
from einops import einsum, reduce, rearrange

class BigramLanguageModel(nn.Module):
    """
    Class definition for a simple bigram language model.
    """
    def __init__(self, config):
        """
        Initialize the bigram language model.

        The model should have the following layers:
        1. An embedding layer that maps tokens to embeddings. (self.embeddings)
        2. A linear layer that maps embeddings to logits. (self.linear) **set bias to True**
        3. A dropout layer. (self.dropout)

        NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """
        super(BigramLanguageModel, self).__init__()
        # Define layers as specified
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embed_dim
        self.dropout_prob = config.dropout
        # Initialize embedding layer
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)

        # Initialize linear layer
        self.linear = nn.Linear(self.embedding_dim, self.vocab_size, bias=True)

        # Initialize dropout layer
        self.dropout = nn.Dropout(self.dropout_prob)

        self.apply(self._init_weights)

    def forward(self, x):
        """
        Forward pass of the bigram language model.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, 2) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, vocab_size) containing the logits.
        """
        # Get embeddings for input tokens
        x = self.embeddings(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Linear layer to get logits
        logits = self.linear(x)
        
        return logits

    def _init_weights(self, module):
        """
        Weight initialization for better convergence.

        NOTE : You do not need to modify this function.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate(self, context, max_new_tokens=100):
        """
        Use the model to generate new tokens given a context.
        We will perform multinomial sampling which is very similar to greedy sampling
        but instead of taking the token with the highest probability, we sample the next token from a multinomial distribution.

        Args:
        context : List[int]
            A list of integers (tokens) representing the context.
        max_new_tokens : int
            The maximum number of new tokens to generate.

        Output:
        List[int]
            A list of integers (tokens) representing the generated tokens.
        """
        context = torch.tensor(context).unsqueeze(0)  # Make it (1, len(context))
        generated_tokens = context.tolist()[0]

        for _ in range(max_new_tokens):
            logits = self.forward(context)  # Get logits for the current context
            logits = logits[:, -1, :]  # Get logits of the last token
            probabilities = torch.softmax(logits, dim=-1)  # Convert to probabilities
            next_token = torch.multinomial(probabilities, num_samples=1)  # Sample the next token
            context = torch.cat((context, next_token), dim=1)  # Append to the context
            generated_tokens.append(next_token.item())

        return generated_tokens

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SingleHeadAttention(nn.Module):
    """
    Class definition for Single Head Causal Self Attention Layer.
    As in Attention is All You Need (https://arxiv.org/pdf/1706.03762)
    """

    def __init__(self, input_dim, output_key_query_dim=None, output_value_dim=None, dropout=0.1, max_len=512):
        """
        Initialize the Single Head Attention Layer.

        The model should have the following layers:
        1. A linear layer for key. (self.key) **set bias to False**
        2. A linear layer for query. (self.query) **set bias to False**
        3. A linear layer for value. (self.value) # **set bias to False**
        4. A dropout layer. (self.dropout)
        5. A causal mask. (self.causal_mask) This should be registered as a buffer.

         NOTE : PLEASE KEEP THE NAME OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_key_query_dim = output_key_query_dim or input_dim
        self.output_value_dim = output_value_dim or input_dim

        self.key = nn.Linear(input_dim, self.output_key_query_dim, bias=False)
        self.query = nn.Linear(input_dim, self.output_key_query_dim, bias=False)
        self.value = nn.Linear(input_dim, self.output_value_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        causal_mask = torch.triu(torch.ones(max_len, max_len),diagonal=1).bool().unsqueeze(0)
        self.register_buffer("causal_mask", causal_mask)

        

    def forward(self, x):
        """
        Forward pass of the Single Head Attention Layer.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, output_value_dim) containing the output tokens.
        """
        batch_size, num_tokens, _ = x.size()
        keys = self.key(x)  # (batch_size, num_tokens, output_key_query_dim)
        queries = self.query(x)  # (batch_size, num_tokens, output_key_query_dim)
        values = self.value(x)  # (batch_size, num_tokens, output_value_dim)

        # Calculate attention scores
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(self.output_key_query_dim)

        # Apply causal mask
        attention_scores = attention_scores.masked_fill(self.causal_mask[ :, :num_tokens, :num_tokens] , float('-inf'))

        # Apply softmax to get attention weights
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Ensure attention_probs has the correct shape for batch matrix multiplication
        # attention_probs should be (batch_size, num_tokens, num_tokens)
          # Remove the extra dimension

        # Compute context vector
        context = torch.bmm(attention_probs, values)  # (batch_size, num_tokens, output_value_dim)

        return context




class MultiHeadAttention(nn.Module):
    """
    Class definition for Multi Head Attention Layer.
    As in Attention is All You Need (https://arxiv.org/pdf/1706.03762)
    """

    def __init__(self, input_dim, num_heads, dropout=0.1) -> None:
        """
        Initialize the Multi Head Attention Layer.

        The model should have the following layers:
        1. Multiple SingleHeadAttention layers. (self.head_{i}) Use setattr to dynamically set the layers.
        2. A linear layer for output. (self.out) **set bias to True**
        3. A dropout layer. (self.dropout) Apply dropout to the output of the out layer.

        NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads

        for i in range(num_heads):
            setattr(self, f'head_{i}', SingleHeadAttention(input_dim, input_dim // num_heads, input_dim // num_heads, dropout))

        self.out = nn.Linear(input_dim, input_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of the Multi Head Attention Layer.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the output tokens.
        """
        head_outputs = [getattr(self, f'head_{i}')(x) for i in range(self.num_heads)]
        concat_heads = torch.cat(head_outputs, dim=-1)
        output = self.out(concat_heads)
        output = self.dropout(output)
        return output


class FeedForwardLayer(nn.Module):
    """
    Class definition for Feed Forward Layer.
    """

    def __init__(self, input_dim, feedforward_dim=None, dropout=0.1):
        """
        Initialize the Feed Forward Layer.

        The model should have the following layers:
        1. A linear layer for the feedforward network. (self.fc1) **set bias to True**
        2. A GELU activation function. (self.activation)
        3. A linear layer for the feedforward network. (self.fc2) ** set bias to True**
        4. A dropout layer. (self.dropout)

        NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """
        super().__init__()

        feedforward_dim = feedforward_dim or input_dim * 4

        self.fc1 = nn.Linear(input_dim, feedforward_dim, bias=True)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(feedforward_dim, input_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of the Feed Forward Layer.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the output tokens.
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    """
    LayerNorm module as in the paper https://arxiv.org/abs/1607.06450

    Note : Variance computation is done with biased variance.
    """

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True) -> None:
        super().__init__()

        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(tuple(self.normalized_shape)))
            self.beta = nn.Parameter(torch.zeros(tuple(self.normalized_shape)))

    def forward(self, input):
        """
        Forward pass of the LayerNorm Layer.

        Args:
        input : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the output tokens.
        """
        mean = input.mean(dim=-1, keepdim=True)
        std = input.std(dim=-1, keepdim=True, unbiased=False)
        normalized_input = (input - mean) / (std + self.eps)

        if self.elementwise_affine:
            normalized_input = self.gamma * normalized_input + self.beta

        return normalized_input


class TransformerLayer(nn.Module):
    """
    Class definition for a single transformer layer.
    """

    def __init__(self, input_dim, num_heads, feedforward_dim=None):
        super().__init__()
        """
        Initialize the Transformer Layer.
        We will use prenorm layer where we normalize the input before applying the attention and feedforward layers.

        The model should have the following layers:
        1. A LayerNorm layer. (self.norm1)
        2. A MultiHeadAttention layer. (self.attention)
        3. A LayerNorm layer. (self.norm2)
        4. A FeedForwardLayer layer. (self.feedforward)

        NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """
        self.norm1 = LayerNorm(input_dim)
        self.attention = MultiHeadAttention(input_dim, num_heads)
        self.norm2 = LayerNorm(input_dim)
        self.feedforward = FeedForwardLayer(input_dim, feedforward_dim)

    def forward(self, x):
        """
        Forward pass of the Transformer Layer.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the output tokens.
        """
        x = x + self.attention(self.norm1(x))
        x = x + self.feedforward(self.norm2(x))
        return x


class MiniGPT(nn.Module):
    """
    Putting it all together: GPT model
    """

    def __init__(self, config) -> None:
        super().__init__()
        """
        Putting it all together: our own GPT model!

        Initialize the MiniGPT model.

        The model should have the following layers:
        1. An embedding layer that maps tokens to embeddings. (self.vocab_embedding)
        2. A positional embedding layer. (self.positional_embedding) We will use learnt positional embeddings.
        3. A dropout layer for embeddings. (self.embed_dropout)
        4. Multiple TransformerLayer layers. (self.transformer_layers)
        5. A LayerNorm layer before the final layer. (self.prehead_norm)
        6. Final language Modelling head layer. (self.head) We will use weight tying (https://paperswithcode.com/method/weight-tying) and set the weights of the head layer to be the same as the vocab_embedding layer.

        NOTE: You do not need to modify anything here.
        """
        self.vocab_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.positional_embedding = nn.Embedding(config.context_length, config.embed_dim)
        self.embed_dropout = nn.Dropout(config.embed_dropout)

        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(config.embed_dim, config.num_heads, config.feedforward_size)
             for _ in range(config.num_layers)]
        )

        # prehead layer norm
        self.prehead_norm = LayerNorm(config.embed_dim)

        self.head = nn.Linear(config.embed_dim, config.vocab_size)  # Language modelling head

        if config.weight_tie:
            self.head.weight = self.vocab_embedding.weight

        # precreate positional indices for the positional embedding
        pos = torch.arange(0, config.context_length, dtype=torch.long)
        self.register_buffer("pos", pos, persistent=False)

        self.apply(self._init_weights)

    def forward(self, x):
        """
        Forward pass of the MiniGPT model.

        Remember to add the positional embeddings to your input token!!

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, seq_len) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, seq_len, vocab_size) containing the logits.
        """
        seq_len = x.size(1)
        token_embeddings = self.vocab_embedding(x)
        position_embeddings = self.positional_embedding(self.pos[:seq_len])
        x = token_embeddings + position_embeddings
        x = self.embed_dropout(x)

        for layer in self.transformer_layers:
            x = layer(x)

        x = self.prehead_norm(x)
        logits = self.head(x)
        return logits

    def _init_weights(self, module):
        """
        Weight initialization for better convergence.

        NOTE : You do not need to modify this function.
        """
        if isinstance(module, nn.Linear):
            if module._get_name() == "fc2":
                # GPT-2 style FFN init
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate(self, context, max_new_tokens=100):
        """
        Use the model to generate new tokens given a context.

        Please copy the generate function from the BigramLanguageModel class you had implemented earlier.
        """
        generated = context
        context_tensor = torch.tensor(context).unsqueeze(0)

        # for _ in range(max_new_tokens):
        #     logits = self(context_tensor)[:, -1, :]
        #     probs = torch.nn.functional.softmax(logits, dim=-1)
        #     next_token = torch.multinomial(probs, num_samples=1).item()
        #     generated.append(next_token)
        #     context_tensor = torch.tensor(generated).unsqueeze(0)

        # return generated
        for _ in range(max_new_tokens):
            # Crop the context if needed
            context_condensed = context if context.size(1) <= self.pos.size(0) else context[:, -self.pos.size(0):]
    
            # Get the logits from the model
            logits = self(context_condensed)
    
            # Focus only on the last time step
            logits = logits[:, -1, :]  # (batch_size, vocab_size)
    
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)  # (batch_size, vocab_size)
    
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
    
            # Append the sampled token to the context
            context = torch.cat((context, next_token), dim=1)  # (batch_size, seq_len+1)
    
        return context
