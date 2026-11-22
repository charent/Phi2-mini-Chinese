"""路径解析与配置默认值。"""

from __future__ import annotations

from phi2_mini import paths
from phi2_mini.config import CustomArugments, TrainConfig


def test_project_root_points_to_repo():
    assert (paths.PROJECT_ROOT / 'pyproject.toml').is_file()


def test_default_dirs_live_under_project_root():
    for directory in (paths.CONFIG_DIR, paths.DATA_DIR, paths.MODEL_DIR, paths.LOG_DIR, paths.CACHE_DIR):
        assert directory.parent == paths.PROJECT_ROOT


def test_resolve_path_prefers_explicit_value():
    assert paths.resolve_path('custom/dir', paths.DATA_DIR) == 'custom/dir'


def test_resolve_path_falls_back_to_default():
    assert paths.resolve_path(None, paths.DATA_DIR) == str(paths.DATA_DIR)


def test_train_config_defaults():
    config = TrainConfig()
    assert config.train_type is None
    assert config.train_from_model_dir is None
    assert config.train_files is None
    assert config.max_seq_len == 512
    assert config.num_proc == 8
    assert config.logs_dir == 'logs'
    assert config.cache_dir is None


def test_legacy_alias_points_to_new_class():
    assert CustomArugments is TrainConfig
