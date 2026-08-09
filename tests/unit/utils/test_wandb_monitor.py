from prime_rl.configs.trainer import BaseOptimizerConfig
from prime_rl.utils.monitor.wandb import _run_config_payload


def test_shared_run_config_is_namespaced_by_writer_label():
    run_config = BaseOptimizerConfig(lr=1e-6)

    payload = _run_config_payload(run_config, shared_mode=True, label="trainer")

    assert payload == {
        "trainer": {
            "lr": 1e-6,
            "weight_decay": 0.01,
            "max_norm": 1.0,
        }
    }


def test_non_shared_run_config_remains_unnamespaced():
    run_config = BaseOptimizerConfig(lr=1e-6)

    payload = _run_config_payload(run_config, shared_mode=False, label=None)

    assert payload == {
        "lr": 1e-6,
        "weight_decay": 0.01,
        "max_norm": 1.0,
    }
