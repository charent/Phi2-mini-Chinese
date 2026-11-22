"""预训练 collator 的输出形状与 mask 语义。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip('torch')

from phi2_mini.data.collator import EfficiencyMaskCollator  # noqa: E402


def test_batch_shapes_and_labels(stub_tokenizer):
    collator = EfficiencyMaskCollator(stub_tokenizer)
    batch = collator([{'input_ids': [10, 11, 2, 20]}, {'input_ids': [21, 2, 10, 2]}])

    assert batch['input_ids'].shape == (2, 4)
    assert batch['attention_mask'].shape == (2, 1, 4, 4)
    assert torch.equal(batch['labels'], batch['input_ids'])


def test_padding_positions_are_masked(stub_tokenizer):
    collator = EfficiencyMaskCollator(stub_tokenizer)
    batch = collator([{'input_ids': [10, 2, 11]}, {'input_ids': [21, 2]}])

    assert batch['input_ids'][1].tolist() == [21, 2, 0]
    assert batch['labels'][1].tolist() == [21, 2, -100]

    mask = batch['attention_mask'][1, 0]
    # padding 位置不能看真实 token，也不能被真实 token 看到
    assert mask[:2, -1].sum().item() == 0
    assert mask[-1, :2].sum().item() == 0
    # 但 padding 行保留对角线，避免整行被屏蔽导致 softmax NaN
    assert mask[-1, -1].item() == 1


def test_different_docs_cannot_see_each_other(stub_tokenizer):
    collator = EfficiencyMaskCollator(stub_tokenizer)
    # 一个样本内合并了两个 doc：[10, 2] 与 [20, 21, 2]
    batch = collator([{'input_ids': [10, 2, 20, 21, 2]}])

    mask = batch['attention_mask'][0, 0]
    assert mask[:2, 2:].sum().item() == 0
    assert mask[2:, :2].sum().item() == 0
    # 同 doc 内仍是因果 mask
    assert mask[0, 1].item() == 0
    assert mask[1, 0].item() == 1


def test_pad_to_multiple_of(stub_tokenizer):
    collator = EfficiencyMaskCollator(stub_tokenizer, pad_to_multiple_of=8)
    batch = collator([{'input_ids': [10, 11, 2]}])
    assert batch['input_ids'].shape == (1, 8)


def test_rejects_non_pt_tensors(stub_tokenizer):
    collator = EfficiencyMaskCollator(stub_tokenizer)
    with pytest.raises(ValueError, match='return_tensors'):
        collator([{'input_ids': [10, 2]}], return_tensors='np')


def test_rejects_empty_features(stub_tokenizer):
    collator = EfficiencyMaskCollator(stub_tokenizer)
    with pytest.raises(ValueError, match='features'):
        collator([])
