import pytest
import verifiers.v1 as vf

from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.transport import TrainingSample


def _node(
    *,
    parent: int | None,
    sampled: bool,
    token_ids: list[int],
    mask: list[bool],
    logprobs: list[float],
) -> vf.MessageNode:
    message = vf.AssistantMessage(content="action") if sampled else vf.UserMessage(content="context")
    return vf.MessageNode.model_construct(
        parent=parent,
        message=message,
        sampled=sampled,
        token_ids=token_ids,
        mask=mask,
        logprobs=logprobs,
    )


def _trace(nodes: list[vf.MessageNode]) -> vf.Trace:
    return vf.Trace.model_construct(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None)),
        nodes=nodes,
    )


@pytest.mark.parametrize(
    ("token_ids", "mask", "logprobs", "message"),
    [
        ([1, 2], [True], [-0.1], "2 token ids but 1 mask values"),
        ([1], [1], [-0.1], "mask values must be exact bools"),
        ([1, 2], [True, True], [-0.1], "2 completion tokens but 1 logprobs"),
        ([1, 2], [True, True], [-0.1, -0.2, -0.3], "2 completion tokens but 3 logprobs"),
    ],
)
def test_trace_to_samples_rejects_malformed_node_streams(
    token_ids,
    mask,
    logprobs,
    message,
):
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


def test_trace_to_samples_preserves_nested_branches():
    trace = _trace(
        [
            _node(parent=None, sampled=False, token_ids=[1], mask=[False], logprobs=[]),
            _node(parent=0, sampled=True, token_ids=[2], mask=[True], logprobs=[-0.2]),
            _node(parent=1, sampled=False, token_ids=[3], mask=[False], logprobs=[]),
            _node(parent=2, sampled=True, token_ids=[4], mask=[True], logprobs=[-0.4]),
            _node(parent=1, sampled=False, token_ids=[5], mask=[False], logprobs=[]),
            _node(parent=4, sampled=True, token_ids=[6], mask=[True], logprobs=[-0.6]),
        ]
    )

    samples = trace_to_samples(trace, env_name="test-env")

    if not all(isinstance(sample, TrainingSample) for sample in samples):
        pytest.fail(f"unexpected samples: {samples!r}")
    actual = [(sample.token_ids, sample.mask, sample.logprobs) for sample in samples]
    expected = [
        ([1, 2, 3, 4], [False, True, False, True], [0.0, -0.2, 0.0, -0.4]),
        ([1, 2, 5, 6], [False, False, False, True], [0.0, -0.2, 0.0, -0.6]),
    ]
    if actual != expected:
        pytest.fail(f"nested branches changed to {actual!r}")
