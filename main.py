import math
import torch
from torch import nn
from torch.nn import functional as F

CHUNK_SIZE = 8
BATCH_SIZE = 32


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()

        self.token_embedding_table = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=vocab_size,
        )

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        logits = self.token_embedding_table(idx)

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
            logits, _loss = self(idx) # outputs [batch_size, sequence_length, channels]

            # Focus on only logits from the last time step
            logits = logits[:, -1, :] # becomes [batch_size, channels]

            # Get probability distribution using softmax
            probs = F.softmax(logits, dim=-1)

            # Sample from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1) # [batch_size, 1]

            # Append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # [batch_size, sequence_length + 1]

        return idx


def get_vocabulary(dataset: str) -> list[str]:
    """Using a simple character-based vocabulary for now."""
    return list(sorted(set(dataset)))


def get_encoder_decoder(vocabulary: list[str]) -> tuple[callable, callable]:    
    encoding_scheme = { vocab: index for index, vocab in enumerate(vocabulary) }
    decoding_scheme = { index: vocab for index, vocab in enumerate(vocabulary) }
    encode = lambda text: [encoding_scheme[character] for character in text]
    decode = lambda encoding: [decoding_scheme[code] for code in encoding]
    return encode, decode


def get_batch(dataset: list[int], chunk_size: int, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    sample_indices = torch.randint(len(dataset) - chunk_size, (batch_size,))
    x = torch.stack([dataset[i:i+chunk_size] for i in sample_indices]).to(device)
    y = torch.stack([dataset[i+1:i+chunk_size+1] for i in sample_indices]).to(device)
    return x, y


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Using GPU...")
    else:
        device = torch.device("cpu")
        print("GPU is not available. Using CPU...")

    with open("./dataset.txt", "r") as dataset_file:
        dataset = dataset_file.read()

    vocabulary = get_vocabulary(dataset=dataset)
    print(f"Vocabulary size: {len(vocabulary)}")

    encode, decode = get_encoder_decoder(vocabulary=vocabulary)

    encoded_dataset = torch.tensor(encode(dataset), dtype=torch.long)

    train_test_split = math.floor(len(encoded_dataset) * 0.9)
    train_set = encoded_dataset[:train_test_split]
    test_set = encoded_dataset[train_test_split:]

    bigram_model = BigramLanguageModel(vocab_size=len(vocabulary)).to(device)

    optimizer = torch.optim.AdamW(bigram_model.parameters(), lr=1e-3)
    for _ in range(10000):
        xb, yb = get_batch(dataset=train_set, chunk_size=CHUNK_SIZE, batch_size=BATCH_SIZE, device=device)

        _logits, loss = bigram_model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        print(loss.item())

    print(''.join(decode(bigram_model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=1000)[0].tolist())))


if __name__ == "__main__":
    main()

