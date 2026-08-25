STOI = {chr(i): i for i in range(256)}
ITOS = {i: chr(i) for i in range(256)}
EOT = 0
VOCAB_SIZE = len(STOI)
def encode(src: str) -> list[int]:
    return [STOI[c] for c in src]
def decode(src: list[int]) -> str:
    return "".join(ITOS[t] for t in src)