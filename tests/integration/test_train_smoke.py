"""训练冒烟测试：微型模型 + 微型数据跑 1 step，验证整条链路没有断。

默认被 ``-m "not slow"`` 跳过；本地用 ``make smoke`` 或 ``pytest -m slow`` 执行。
"""

from __future__ import annotations

import pytest

pytest.importorskip('torch')
pytest.importorskip('datasets')
pytest.importorskip('accelerate')
pytest.importorskip('transformers')

from phi2_mini.config import TrainConfig
from phi2_mini.modeling.factory import build_phi_config


@pytest.mark.slow
def test_pretrain_runs_one_step(tmp_path, monkeypatch, tiny_tokenizer):
    from datasets import Dataset
    from transformers import TrainingArguments

    from phi2_mini.data import pretrain as data_pretrain
    from phi2_mini.train import pretrain as train_pretrain

    tokenizer_dir = tmp_path / 'tokenizer'
    tiny_tokenizer.save_pretrained(tokenizer_dir)

    raw = Dataset.from_dict({'text': ['你好 世界', '感冒 你好', '世界 感冒', '你好 感冒'] * 4})
    monkeypatch.setattr(data_pretrain, 'load_parquet_dataset', lambda *args, **kwargs: raw)

    def tiny_phi_config(tokenizer, attn_implementation):
        return build_phi_config(
            tokenizer,
            attn_implementation,
            hidden_size=32,
            num_attention_heads=2,
            num_hidden_layers=2,
            intermediate_size=64,
            max_position_embeddings=64,
        )

    monkeypatch.setattr(train_pretrain, 'build_phi_config', tiny_phi_config)
    monkeypatch.setattr(train_pretrain, 'resolve_attn_implementation', lambda *args, **kwargs: 'eager')

    config = TrainConfig(
        train_type='pre',
        train_files=['unused.parquet'],
        tokenizer_dir=str(tokenizer_dir),
        max_seq_len=16,
        logs_dir=str(tmp_path / 'logs'),
        num_proc=1,
        cache_dir=str(tmp_path / 'cache'),
    )
    train_args = TrainingArguments(
        output_dir=str(tmp_path / 'out'),
        max_steps=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_strategy='no',
        report_to=[],
        use_cpu=True,
        remove_unused_columns=False,
    )

    train_pretrain.pre_train(config, train_args)

    assert (tmp_path / 'out' / 'config.json').is_file()
    assert (tmp_path / 'out' / 'model.safetensors').is_file()
    assert list((tmp_path / 'logs').glob('pre_train_*.csv'))
