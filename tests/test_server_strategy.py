import numpy as np
import flwr as fl
import flwr.server.client_proxy as cp

from server import get_strategy


class DummyProxy(cp.ClientProxy):
    def get_properties(self, ins, timeout, group_id): pass
    def get_parameters(self, ins, timeout, group_id): pass
    def fit(self, ins, timeout, group_id): pass
    def evaluate(self, ins, timeout, group_id): pass
    def reconnect(self, ins, timeout, group_id): pass


def _fit_res(weight_value, num_examples=100):
    arrays = [np.full((4, 4), weight_value, dtype=np.float32)]
    return fl.common.FitRes(
        status=fl.common.Status(code=fl.common.Code.OK, message="ok"),
        parameters=fl.common.ndarrays_to_parameters(arrays),
        num_examples=num_examples,
        metrics={"epsilon": 1.0},
    )


def test_trust_aware_aggregation_excludes_poisoned_outlier():
    """
    Simulates a compromised client (id 5) submitting a wildly out-of-range
    update norm, as in the gradient-poisoning threat model. It should be
    dropped from aggregation while the five well-behaved clients are kept.
    """
    strategy = get_strategy()
    results = [
        (DummyProxy(cid="0"), _fit_res(1.0)),
        (DummyProxy(cid="1"), _fit_res(1.05)),
        (DummyProxy(cid="2"), _fit_res(0.95)),
        (DummyProxy(cid="3"), _fit_res(1.02)),
        (DummyProxy(cid="4"), _fit_res(1.01)),
        (DummyProxy(cid="5"), _fit_res(500.0)),
    ]

    _, metrics = strategy.aggregate_fit(server_round=1, results=results, failures=[])

    assert metrics["trusted_clients"] == 5


def test_trust_aware_aggregation_keeps_all_when_updates_are_uniform():
    strategy = get_strategy()
    results = [(DummyProxy(cid=str(i)), _fit_res(1.0)) for i in range(6)]

    _, metrics = strategy.aggregate_fit(server_round=1, results=results, failures=[])

    assert metrics["trusted_clients"] == 6
