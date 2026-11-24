"""Tests for the embedding similarity command."""

import pytest

from scripts import compare


def test_cosine_similarity():
    assert compare.cosine_similarity([1, 0], [1, 0]) == pytest.approx(1)
    assert compare.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0)
    assert compare.cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1)


@pytest.mark.parametrize('a,b', [([0, 0], [1, 1]), ([1], [1, 2]), ([], [])])
def test_invalid_vectors_are_rejected(a, b):
    with pytest.raises(ValueError):
        compare.cosine_similarity(a, b)


def test_cli_uses_embedder_and_prints_similarity(monkeypatch, capsys):
    class FakeEmbedder:
        def embed_one(self, text):
            return {'first': [1.0, 0.0], 'second': [1.0, 1.0]}[text]

    monkeypatch.setattr(compare, 'OpenAIEmbedder', FakeEmbedder)
    monkeypatch.setattr(compare, 'load_dotenv', lambda *args, **kwargs: None)
    assert compare.main(['--text-a', 'first', '--text-b', 'second']) == 0
    output = capsys.readouterr().out
    assert 'Text A: first' in output
    assert 'Text B: second' in output
    assert 'Cosine similarity: 0.7071' in output
