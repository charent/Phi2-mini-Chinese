.DEFAULT_GOAL := help
.PHONY: help install fmt lint format type test cov smoke pre-commit demo

help:
	@echo "install    安装训练/开发依赖（uv sync --extra train）"
	@echo "fmt        ruff format"
	@echo "lint       ruff check --fix"
	@echo "format     fmt + lint"
	@echo "type       pyright（目标 0 error）"
	@echo "test       单元测试（跳过 slow/gpu）"
	@echo "cov        单元测试 + 覆盖率"
	@echo "smoke      冒烟测试（slow 用例：1 step 训练）"
	@echo "pre-commit 对全部文件跑 pre-commit"
	@echo "demo       启动 gradio 对话 demo"

install:
	uv sync --extra train

fmt:
	uv run ruff format .

lint:
	uv run ruff check --fix .

format: fmt lint

type:
	uv run pyright

test:
	uv run pytest -m "not slow and not gpu"

cov:
	uv run pytest -m "not slow and not gpu" --cov=phi2_mini --cov-report=term-missing

smoke:
	uv run pytest -m slow

pre-commit:
	uv run pre-commit run --all-files

demo:
	uv run python -m phi2_mini.inference.web_demo
