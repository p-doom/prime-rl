import tomllib
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from prime_rl.configs.inference import ServerConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.entrypoints.rl import (
    configure_inference_advertisement,
    resolve_inference_advertise_host,
    rl_local,
)


def _config(
    *,
    advertise_host: str | None,
    bind_host: str | None = "inference-a",
    base_url: list[str] | None = None,
    admin_base_url: list[str] | None = None,
):
    server = SimpleNamespace(host=bind_host, advertise_host=advertise_host)
    inference = SimpleNamespace(server=server)
    client = SimpleNamespace(
        base_url=base_url or ["http://localhost:8000/v1"],
        admin_base_url=admin_base_url,
    )
    orchestrator = SimpleNamespace(model=SimpleNamespace(client=client))
    return SimpleNamespace(inference=inference, orchestrator=orchestrator)


def test_advertisement_is_opt_in():
    config = _config(advertise_host=None)

    assert not configure_inference_advertisement(config)
    assert config.orchestrator.model.client.base_url == ["http://localhost:8000/v1"]


def test_advertise_host_rejects_explicit_values():
    with pytest.raises(ValidationError):
        ServerConfig(advertise_host="inference-a")


def test_auto_advertisement_uses_allocated_slurm_node_for_wildcard_bind():
    assert (
        resolve_inference_advertise_host(
            "0.0.0.0",
            environ={"SLURMD_NODENAME": "inference-a"},
        )
        == "inference-a"
    )


def test_auto_advertisement_falls_back_to_system_hostname():
    assert (
        resolve_inference_advertise_host(
            "::",
            environ={},
            hostname_func=lambda: "inference-a",
        )
        == "inference-a"
    )


def test_auto_advertisement_uses_concrete_bind_host():
    assert resolve_inference_advertise_host("10.0.0.4") == "10.0.0.4"


@pytest.mark.parametrize("bind_host", ["localhost", "localhost.", "127.0.0.2", "0:0:0:0:0:0:0:1"])
def test_advertisement_rejects_loopback_bind(bind_host):
    with pytest.raises(ValueError, match="loopback-only"):
        resolve_inference_advertise_host(bind_host)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost.:8000/v1",
        "http://127.0.0.2:8000/v1",
        "http://0.0.0.0:8000/v1",
        "http://[::]:8000/v1",
        "http://[0:0:0:0:0:0:0:1]:8000/v1",
    ],
)
def test_advertisement_replaces_all_local_only_url_forms(base_url):
    config = _config(advertise_host="auto", base_url=[base_url])

    assert configure_inference_advertisement(config)
    assert config.orchestrator.model.client.base_url == ["http://inference-a:8000/v1"]
    assert config.orchestrator.model.client.admin_base_url == [base_url]


def test_advertisement_preserves_remote_and_explicit_admin_urls():
    config = _config(
        advertise_host="auto",
        base_url=["http://localhost:8000/v1", "http://inference-b:8000/v1"],
        admin_base_url=["http://admin:9000/v1"],
    )

    assert configure_inference_advertisement(config)
    assert config.orchestrator.model.client.base_url == [
        "http://inference-a:8000/v1",
        "http://inference-b:8000/v1",
    ]
    assert config.orchestrator.model.client.admin_base_url == ["http://admin:9000/v1"]


def test_local_launcher_serializes_allocated_host_when_enabled(tmp_path, monkeypatch):
    config = RLConfig.model_validate(
        {
            "output_dir": tmp_path,
            "dry_run": True,
            "trainer": {},
            "orchestrator": {},
            "inference": {"server": {"advertise_host": "auto"}},
            "deployment": {
                "type": "single_node",
                "gpus_per_node": 2,
                "num_train_gpus": 1,
                "num_infer_gpus": 1,
            },
        }
    )
    monkeypatch.setenv("SLURMD_NODENAME", "inference-a")

    rl_local(config)

    with (tmp_path / "configs" / "orchestrator.toml").open("rb") as f:
        orchestrator = tomllib.load(f)
    client = orchestrator["model"]["client"]
    assert client["base_url"] == ["http://inference-a:8000/v1"]
    assert client["admin_base_url"] == ["http://localhost:8000/v1"]
