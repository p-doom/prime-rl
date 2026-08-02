from unittest.mock import Mock

import prime_rl.utils.monitor.wandb as wandb_monitor


def test_list_views_uses_compatible_graphql_client(monkeypatch):
    api = object()
    execute_graphql = Mock(
        return_value={"project": {"allViews": {"edges": [{"node": {"displayName": "Overview", "name": "nw-view-v"}}]}}}
    )
    monkeypatch.setattr(wandb_monitor.wandb, "Api", lambda: api)
    monkeypatch.setattr(wandb_monitor, "execute_graphql", execute_graphql)

    assert wandb_monitor.list_views("entity", "project") == [("Overview", "nw-view-v")]
    execute_graphql.assert_called_once()
    call = execute_graphql.call_args
    assert call.args[0] is api
    assert "query Views" in call.args[1]
    assert call.args[2] == {"entity": "entity", "project": "project"}
