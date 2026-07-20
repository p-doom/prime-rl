import shlex

from prime_rl.utils.uv import (
    PRIME_RL_UV_SYNC_ARGS_ENV,
    shell_quote,
    uv_sync_args_from_env,
)


def test_uv_sync_args_are_parsed_and_shell_quoted(monkeypatch):
    monkeypatch.setenv(
        PRIME_RL_UV_SYNC_ARGS_ENV,
        '--flag "value with spaces" "semi;colon" "single\'quote"',
    )

    args = uv_sync_args_from_env()
    rendered = " ".join(shell_quote(arg) for arg in args)

    assert args == ["--flag", "value with spaces", "semi;colon", "single'quote"]
    assert shlex.split(rendered) == args
