import os
import shlex

PRIME_RL_UV_SYNC_ARGS_ENV = "PRIME_RL_UV_SYNC_ARGS"


def uv_sync_args_from_env() -> list[str]:
    """Parse additional uv sync arguments supplied by the launch environment."""
    return shlex.split(os.environ.get(PRIME_RL_UV_SYNC_ARGS_ENV, ""))


def shell_quote(value: str) -> str:
    """Quote one argument for embedding in a generated shell script."""
    return shlex.quote(value)
