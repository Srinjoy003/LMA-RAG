import hashlib
import re

import numpy as np
import pytest

from fixtures.make_bank import write_bank
from part1.bank import load_bank


def fake_embed(texts):
    """Deterministic bag-of-words embedder: hashes each word into one of 64 slots."""
    out = np.zeros((len(texts), 64), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in re.findall(r"\w+", text.lower()):
            slot = int(hashlib.md5(word.encode()).hexdigest()[:8], 16) % 64
            out[row, slot] += 1.0
    return out


@pytest.fixture
def embed():
    return fake_embed


@pytest.fixture
def bank_dir(tmp_path):
    """A complete, valid bank written to a temp dir."""
    return write_bank(tmp_path / "bank")


@pytest.fixture
def bank(bank_dir):
    return load_bank(bank_dir)
