"""Gradio 流式对话 demo。

用法::

    uv run python examples/web_demo.py
    uv run python -m phi2_mini.inference.web_demo

权重目录优先级：环境变量 ``TOKENIZER_DIR`` / ``MODEL_DIR`` > 本地训练产物 > 开源发布权重。
重构前的版本在 import 时就会加载模型（导致单测无法导入本模块），现在改为 ``main()`` 里加载。
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from threading import Thread
from typing import Any

import gradio as gr
import torch
from transformers import AutoTokenizer, GenerationConfig, TextIteratorStreamer

from phi2_mini.logging_utils import get_logger
from phi2_mini.modeling.factory import load_phi_model
from phi2_mini.paths import MODEL_DIR

logger = get_logger(__name__)

HF_REPO_ID = 'charent/Phi2-Chinese-0.2B'
PROMPT_TEMPLATE = '##提问:\n{message}\n##回答:\n'
DEFAULT_MAX_NEW_TOKENS = 512
SERVER_NAME = '127.0.0.1'
SERVER_PORT = 7860


def pick_dir(env_key: str, candidates: list[str | Path]) -> str:
    """环境变量优先，其次第一个存在的候选目录，最后回落到 HF 仓库 id。"""
    env_value = os.getenv(env_key)
    if env_value:
        return env_value
    for candidate in candidates:
        if Path(candidate).is_dir():
            return str(candidate)
    return HF_REPO_ID


def resolve_dirs() -> tuple[str, str]:
    """返回 ``(tokenizer_dir, model_dir)``。"""
    tokenizer_dir = pick_dir('TOKENIZER_DIR', [MODEL_DIR / 'tokenizer', MODEL_DIR / 'Phi2-Chinese-0.2B'])
    model_dir = pick_dir('MODEL_DIR', [MODEL_DIR / 'dpo', MODEL_DIR / 'Phi2-Chinese-0.2B'])
    return tokenizer_dir, model_dir


def load_model() -> tuple[Any, Any, torch.device]:
    """加载 tokenizer 与模型；返回 ``(tokenizer, model, device)``。"""
    tokenizer_dir, model_dir = resolve_dirs()
    logger.info('tokenizer: %s | model: %s', tokenizer_dir, model_dir)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_phi_model(model_dir)
    # torch 的类型桩把 Module.to 声明成 _Wrapped 描述符，pyright 无法把 device 绑定到 self 之后的参数
    model = model.to(device)  # pyright: ignore[reportArgumentType]
    return tokenizer, model, device


def build_predict(
    tokenizer: Any,
    model: Any,
    device: torch.device,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
):
    """把模型包成 gradio 需要的流式生成器函数。"""
    eos_token = tokenizer.eos_token

    def predict(message: str, history: list) -> Iterator[str]:
        prompt = PROMPT_TEMPLATE.format(message=message)
        model_inputs = tokenizer([prompt], return_tensors='pt', return_token_type_ids=False).to(device)
        streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

        # greedy search
        generation_config = GenerationConfig(
            num_beams=1,
            do_sample=False,
            max_length=320,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        thread = Thread(
            target=model.generate,
            kwargs={**model_inputs, 'streamer': streamer, 'generation_config': generation_config},
        )
        thread.start()

        partial_message = ''
        for new_token in streamer:
            partial_message += new_token
            # 生成到 EOS 就停止
            if eos_token in partial_message:
                break
            yield partial_message

    return predict


def main() -> None:
    """启动 gradio 界面（阻塞）。"""
    tokenizer, model, device = load_model()
    demo = gr.ChatInterface(
        fn=build_predict(tokenizer, model, device),
        title='phi_mini_ChatBot',
        description='phi中文小模型对话测试',
        examples=['你好', '感冒了要怎么办'],
    )
    # predict 是生成器函数，gradio 3.39+ 需要先开启队列才能流式输出
    demo.queue().launch(server_name=SERVER_NAME, server_port=SERVER_PORT)


if __name__ == '__main__':
    main()
