"""pytest 公共 fixture：全部基于假 tokenizer / 微型模型，不下载权重、不依赖 GPU。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:  # 未安装（editable）时也能直接跑 pytest
    sys.path.insert(0, str(SRC))


class StubTokenizer:
    """只实现本仓库用到的 tokenizer 接口，避免单测依赖真实分词器。"""

    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    eos_token = '</s>'

    def __init__(self, vocab: dict[str, list[int]] | None = None, vocab_size: int = 64) -> None:
        self.vocab = vocab or {}
        self.vocab_size = vocab_size

    def __call__(self, texts, **kwargs) -> dict[str, list[list[int]]]:
        return {'input_ids': [self.vocab[text] for text in texts]}

    def encode(self, text: str, **kwargs) -> list[int]:
        return self.vocab[text]

    def __len__(self) -> int:
        return self.vocab_size


@pytest.fixture
def stub_tokenizer() -> StubTokenizer:
    """假 tokenizer：a/b 是普通文本，##提问:/##回答: 是 SFT 模板。"""
    return StubTokenizer(
        {
            'a': [10, 11, 12],
            'b': [20, 21],
            '##提问:': [30],
            '##回答:': [31],
        }
    )


@pytest.fixture
def tiny_tokenizer(tmp_path):
    """真实但极小的 WordLevel tokenizer（用于 smoke test / collator 集成）。"""
    pytest.importorskip('transformers')
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {
        '<pad>': 0,
        '<s>': 1,
        '</s>': 2,
        '你好': 3,
        '世界': 4,
        '感冒': 5,
        '##提问:': 6,
        '##回答:': 7,
    }
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token='<pad>'))
    # tokenizers 的类型桩把 pre_tokenizer 声明为只读 property，运行时是可写的
    backend.pre_tokenizer = pre_tokenizers.Whitespace()  # pyright: ignore[reportAttributeAccessIssue]
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token='<s>',
        eos_token='</s>',
        pad_token='<pad>',
    )
