import pytest
import verifiers.v1 as vf

from prime_rl.orchestrator.trajectories import trace_to_samples


def _node(
    *,
    parent: int | None,
    sampled: bool,
    token_ids: list[int],
    mask: list[bool],
    logprobs: list[float],
) -> vf.MessageNode:
    message = vf.AssistantMessage(content="action") if sampled else vf.UserMessage(content="context")
    return vf.MessageNode(
        parent=parent,
        message=message,
        sampled=sampled,
        token_ids=token_ids,
        mask=mask,
        logprobs=logprobs,
    )


def _trace(nodes: list[vf.MessageNode]) -> vf.Trace:
    return vf.Trace(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None)),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=nodes,
    )


@pytest.mark.parametrize(
    ("token_ids", "mask", "logprobs", "message"),
    [
        ([1, 2], [True], [-0.1], "2 token ids but 1 mask values"),
        ([1, 2], [True, True], [-0.1], "2 sampled tokens but 1 logprobs"),
        ([1, 2], [True, True], [-0.1, -0.2, -0.3], "2 sampled tokens but 3 logprobs"),
    ],
)
def test_trace_to_samples_rejects_malformed_node_streams(token_ids, mask, logprobs, message):
    trace = _trace(
        [
            _node(
                parent=None,
                sampled=True,
                token_ids=token_ids,
                mask=mask,
                logprobs=logprobs,
            )
        ]
    )

    with pytest.raises(ValueError, match=message):
        trace_to_samples(trace, env_name="test-env")


def test_trace_to_samples_associates_logprobs_with_sampled_tokens():
    trace = _trace(
        [
            _node(
                parent=None,
                sampled=True,
                token_ids=[7, 8, 9],
                mask=[True, False, True],
                logprobs=[-0.7, -0.9],
            )
        ]
    )

    samples = trace_to_samples(trace, env_name="test-env")

    actual = [(sample.token_ids, sample.mask, sample.logprobs) for sample in samples]
    expected = [([7, 8, 9], [True, False, True], [-0.7, 0.0, -0.9])]
    if actual != expected:
        pytest.fail(f"logprobs changed token association: {actual!r}")
