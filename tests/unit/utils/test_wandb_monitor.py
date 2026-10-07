from prime_rl.monitors.wandb.monitor import _run_config_payload
from prime_rl.utils.config import BaseConfig


class _OptimizerConfig(BaseConfig):
    lr: float = 1e-6
    max_norm: float = 1.0


def test_shared_run_config_is_namespaced_by_writer_label():
    payload = _run_config_payload(_OptimizerConfig(), shared_mode=True, label="trainer")

    assert payload == {"trainer": {"lr": 1e-6, "max_norm": 1.0}}


def test_non_shared_run_config_remains_unnamespaced():
    payload = _run_config_payload(_OptimizerConfig(), shared_mode=False, label=None)

    assert payload == {"lr": 1e-6, "max_norm": 1.0}
