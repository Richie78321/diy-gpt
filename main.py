import math
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

# Hyperparameters
CHUNK_SIZE = 64
BATCH_SIZE = 32
EVAL_ITERATIONS = 100
EMBEDDING_SIZE = 32
HEAD_SIZE = 16

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("GPU is available. Using GPU...")
else:
    DEVICE = torch.device("cpu")
    print("GPU is not available. Using CPU...")


class FeedForward(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=embedding_size, out_features=embedding_size),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, head_size: int = HEAD_SIZE):
        super().__init__()

        # Bias is False, yielding a simple matrix multiplication with learnable weights
        self.query = nn.Linear(in_features=EMBEDDING_SIZE, out_features=head_size, bias=False)
        self.key = nn.Linear(in_features=EMBEDDING_SIZE, out_features=head_size, bias=False)
        self.value = nn.Linear(in_features=EMBEDDING_SIZE, out_features=head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(CHUNK_SIZE, CHUNK_SIZE)))

    def forward(self, x: torch.Tensor):
        # x is dimension [batch_size, sequence_length, embedding_size]
        _batch_size, sequence_length, embedding_size = x.shape

        # Get query, key, and value matrices
        q = self.query(x) # [batch_size, sequence_length, head_size]
        k = self.key(x) # [batch_size, sequence_length, head_size]
        v = self.value(x) # [batch_size, sequence_length, head_size]

        # Scale the attention weights to maintain an effective variance of 1
        # after computing the dot products.
        attention_weights = q @ k.transpose(-2, -1) * (embedding_size ** -0.5) # [batch_size, sequence_length, head_size] @ [batch_size, head_size, sequence_length] = [batch_size, sequence_length, sequence_length]

        # Mask and normalize the attention weights
        attention_weights = attention_weights.masked_fill(self.tril[:sequence_length, :sequence_length] == 0, float("-inf"))
        attention_weights = F.softmax(attention_weights, dim=-1)

        # Apply attention weights to value matrix
        return attention_weights @ v # [batch_size, sequence_length, sequence_length] @ [batch_size, sequence_length, head_size] = [batch_size, sequence_length, head_size]


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(head_size=head_size) for _ in range(num_heads)])

    def forward(self, x: torch.Tensor):
        # Multi-headed attention is simply using multiple self-attention heads and concatenating
        # the outputs together.
        return torch.cat([head(x) for head in self.heads], dim=-1)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()

        # Intermediate embedding layer
        self.token_embedding_table = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBEDDING_SIZE,
        )

        # Encode the positions of tokens in the sequence
        # These positional embeddings are learned
        self.position_embedding_table = nn.Embedding(
            num_embeddings=CHUNK_SIZE,
            embedding_dim=EMBEDDING_SIZE,
        )

        # For now, the self-attention head size is the same as the embedding size.
        self.msa_head = MultiHeadedAttention(num_heads=4, head_size=EMBEDDING_SIZE // 4)

        self.feedforward = FeedForward(embedding_size=EMBEDDING_SIZE)

        # Final linear layer to get logits
        self.language_model_head = nn.Linear(
            in_features=EMBEDDING_SIZE,
            out_features=vocab_size,
        )

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        # idx [batch_size, sequence_length]
        batch, sequence = idx.shape

        token_embeddings = self.token_embedding_table(idx) # [batch_size, sequence_length, embedding_size]
        positional_embeddings = self.position_embedding_table(torch.arange(sequence, device=DEVICE)) # [sequence_length, embedding_size]
        x = token_embeddings + positional_embeddings # [batch_size, sequence_length, embedding_size]
        x = self.msa_head(x)
        x = self.feedforward(x)
        
        logits = self.language_model_head(x) # [batch_size, sequence_length, vocab_size]

        if targets is None:
            loss = None
        else:
            batch, sequence, channels = logits.shape
            logits = logits.view(batch * sequence, channels)
            targets = targets.view(batch * sequence)
            loss = F.cross_entropy(input=logits, target=targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_context_window = idx[:, -CHUNK_SIZE:]

            logits, _loss = self(idx_context_window) # outputs [batch_size, sequence_length, channels]

            # Focus on only logits from the last time step
            logits = logits[:, -1, :] # becomes [batch_size, channels]

            # Get probability distribution using softmax
            probs = F.softmax(logits, dim=-1)

            # Sample from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1) # [batch_size, 1]

            # Append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # [batch_size, sequence_length + 1]

        return idx


@torch.no_grad()
def estimate_loss(model: nn.Module, train: list[str], test: list[str]):
    model.eval()

    loss_dict: dict[str, float] = {}
    for name, dataset in (("train", train), ("test", test)):
        loss_measurements = torch.zeros(EVAL_ITERATIONS)
        for i in range(EVAL_ITERATIONS):
            bx, by = get_batch(dataset=dataset)
            _out, loss = model(bx, by)
            loss_measurements[i] = loss.item()

        loss_dict[name] = loss_measurements.mean().item()

    model.train()
    return loss_dict


def get_vocabulary(dataset: str) -> list[str]:
    """Using a simple character-based vocabulary for now."""
    return list(sorted(set(dataset)))


def get_encoder_decoder(vocabulary: list[str]) -> tuple[callable, callable]:    
    encoding_scheme = { vocab: index for index, vocab in enumerate(vocabulary) }
    decoding_scheme = { index: vocab for index, vocab in enumerate(vocabulary) }
    encode = lambda text: [encoding_scheme[character] for character in text]
    decode = lambda encoding: [decoding_scheme[code] for code in encoding]
    return encode, decode


def get_batch(dataset: list[int], chunk_size: int=CHUNK_SIZE, batch_size: int=BATCH_SIZE, device: str=DEVICE) -> tuple[torch.Tensor, torch.Tensor]:
    sample_indices = torch.randint(len(dataset) - chunk_size, (batch_size,))
    x = torch.stack([dataset[i:i+chunk_size] for i in sample_indices]).to(device)
    y = torch.stack([dataset[i+1:i+chunk_size+1] for i in sample_indices]).to(device)
    return x, y


def main():
    with open("./dataset.txt", "r") as dataset_file:
        dataset = dataset_file.read()

    vocabulary = get_vocabulary(dataset=dataset)
    print(f"Vocabulary size: {len(vocabulary)}")

    encode, decode = get_encoder_decoder(vocabulary=vocabulary)

    encoded_dataset = torch.tensor(encode(dataset), dtype=torch.long)

    train_test_split = math.floor(len(encoded_dataset) * 0.9)
    train_set = encoded_dataset[:train_test_split]
    test_set = encoded_dataset[train_test_split:]

    bigram_model = BigramLanguageModel(vocab_size=len(vocabulary)).to(DEVICE)

    optimizer = torch.optim.AdamW(bigram_model.parameters(), lr=1e-3)
    for _ in tqdm(range(10_000), desc="Training"):
        xb, yb = get_batch(dataset=train_set)

        _logits, loss = bigram_model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"Final loss: {estimate_loss(model=bigram_model, train=train_set, test=test_set)}")

    print("Test generation:")
    print(''.join(decode(bigram_model.generate(torch.zeros((1, 1), dtype=torch.long, device=DEVICE), max_new_tokens=1000)[0].tolist())))


if __name__ == "__main__":
    main()

