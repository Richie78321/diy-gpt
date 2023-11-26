def get_vocabulary(dataset: str) -> list[str]:
    """Using a simple character-based vocabulary for now."""
    return list(sorted(set(dataset)))


def get_encoder_decoder(vocabulary: list[str]) -> tuple[callable, callable]:    
    encoding_scheme = { vocab: index for index, vocab in enumerate(vocabulary) }
    decoding_scheme = { index: vocab for index, vocab in enumerate(vocabulary) }
    encode = lambda text: [encoding_scheme[character] for character in text]
    decode = lambda encoding: [decoding_scheme[code] for code in encoding]
    return encode, decode


def main():
    with open("./dataset.txt", "r") as dataset_file:
        dataset = dataset_file.read()

    vocabulary = get_vocabulary(dataset=dataset)
    print(f"Vocabulary size: {len(vocabulary)}")

    encode, decode = get_encoder_decoder(vocabulary=vocabulary)

    encoded_dataset = encode(dataset)


if __name__ == "__main__":
    main()

