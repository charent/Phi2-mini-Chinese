# AGENTS.md

写给在本仓库工作的 AI / 人类协作者：环境、目录约定，以及提交前必须跑通的检查。

## 1. 环境与工具

- Python `>=3.10,<3.12`（见 `pyproject.toml`），依赖全部由 **uv** 管理，`uv.lock` 需要提交。
- `uv sync --extra train` 安装训练 + 开发依赖；`uv sync` 只装推理依赖。
- 只用三个检查器：**ruff**（格式 + lint）、**pyright**（类型，目标 0 error）、**pytest**。
  不要引入 mypy / flake8 / black / isort（ruff 已覆盖），也不要用 `uvx`（每次新建临时环境，会刷屏）。

## 2. 目录约定

根目录只允许出现：`pyproject.toml`、`README.md`、`AGENTS.md`、`Makefile`、`LICENSE`、`uv.lock`、
`.gitignore`、`.pre-commit-config.yaml`，以及两个待移除的兼容入口 `train_main.py`、`examples/web_demo.py`。

其余文件按下表落位，**不要在根目录新增脚本或模块**：

| 内容 | 位置 |
| --- | --- |
| 可复用代码、训练流程、数据映射 | `src/phi2_mini/` |
| 数据清洗/去重等可执行脚本 | `src/phi2_mini/data/prepare/`（`python -m phi2_mini.data.prepare.xxx`） |
| notebook | `notebooks/` |
| 训练超参、accelerate 配置 | `configs/` |
| shell / bat 包装 | `scripts/`（只做薄包装，不写业务逻辑） |
| 单元测试 / 集成测试 | `tests/unit/`、`tests/integration/` |
| 数据产物 / 模型产物 / 日志 | `data/`、`model_save/`、`logs/`（均被 ignore） |

## 3. 代码约定

- **类型注解**：公开函数必须有完整的参数与返回注解；不要用 `Any` 兜底。
  `cast(...)` 只用于第三方类型桩不准的地方（如 `from_pretrained` 重载、`MinHashLSH.query`），并在紧邻位置写明原因。
- **路径**：统一 `from phi2_mini.paths import PROJECT_ROOT / DATA_DIR / CONFIG_DIR / LOG_DIR / CACHE_DIR`；
  禁止 `os.getcwd()`、`sys.path.extend(['.', '..'])`、脚本里 `cd ..`。
- **日志**：`from phi2_mini.logging_utils import get_logger`，不要裸 `print`（数据准备进度也走 logger）。
- **重依赖延迟导入**：`transformers` / `datasets` / `trl` / `gradio` 一律在函数内 import，
  保证 `phi2_mini.data.mask`、`phi2_mini.paths` 这类纯逻辑模块在无重型依赖时也能导入和单测。
- **中文文档**：注释/文档正常使用全角标点，ruff 已忽略 `RUF001`–`RUF003`，不要为此改配置。

## 4. 提交前必做（0 error / 0 failure）

```bash
uv run ruff format .
uv run ruff check --fix .
uv run pyright                              # 必须 0 error
uv run pytest -m "not slow and not gpu"     # 必须全绿
```

改到 collator / attention mask / 训练流程时，额外跑一次冒烟训练：

```bash
uv run pytest -m slow
```

也可以用 `make format type test` / `make smoke` 代替。

## 5. 测试约定

- 测试不得下载权重、不得依赖 GPU；用 `tests/conftest.py` 的 `StubTokenizer`（假分词器）
  或 `tiny_tokenizer`（真实但极小的 WordLevel 分词器）。
- 慢用例标 `@pytest.mark.slow`，GPU 用例标 `@pytest.mark.gpu`（`pyproject.toml` 已声明 marker）。
- `create_efficiency_mask` 是预训练行为的锚点：改动 mask 逻辑必须同步
  `tests/unit/test_mask.py` 的 golden 矩阵，以及 `tests/unit/test_collator.py` 的可见性断言。

## 6. 关键实现约束（改动前先读）

- **transformers 固定 4.36.2**：Phi 使用 fused QKV，4.37+ 会拆成 q/k/v 导致发布权重不匹配；
  gradio / fastapi / starlette 版本同样被锁定在注释说明的范围内。
- **预训练 doc packing**：多个短 doc 拼成定长样本后靠 4D 块对角 mask 隔离，而 4.36.2 原生只接受 2D mask。
  因此 `phi2_mini.modeling.packing.patch_phi_attention_mask()` 在模型侧把二进制 mask 转成加性 mask，
  并且**预训练固定 eager**；flash-attn 只在 sft/dpo 生效。
- **DPO 同时加载两份模型**，显存压力大，`EmptyCacheCallback` 不能删。
- 数据去重语义是「保留首次出现」，`n-gram` 默认 3；改默认值会影响已有去重结果。

## 7. 不要做

- 不要恢复 `config.py` / `utils.utils` / `utils.logger` / `train_model.*` / `dataset_prepare.*` 这些旧导入路径。
- 不要为了让检查通过而放宽 `pyproject.toml` 里的 ruff / pyright 规则；确有必要时在该配置项旁写清原因。
- 不要把日志、模型、数据产物写进仓库根目录或 `src/`。
