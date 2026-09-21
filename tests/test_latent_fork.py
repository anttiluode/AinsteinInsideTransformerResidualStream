import unittest

import torch
from torch import nn

from latent_fork import (
    BilinearComposer,
    LinearComposer,
    inject_last_token,
    relative_outside_span,
)


class DummyBlock(nn.Module):
    def forward(self, hidden_states):
        return hidden_states


class DummyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DummyBlock(), DummyBlock()])


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DummyCore()


class ResidueMathTests(unittest.TestCase):
    def test_linear_composer_stays_in_span(self):
        torch.manual_seed(0)
        a = torch.randn(1, 16)
        b = torch.randn(1, 16)
        composer = LinearComposer()
        x = composer(a, b)
        self.assertLess(relative_outside_span(x, a, b).item(), 1e-5)

    def test_bilinear_composer_has_expected_shape(self):
        torch.manual_seed(1)
        a = torch.randn(3, 16)
        b = torch.randn(3, 16)
        composer = BilinearComposer(hidden_size=16, rank=7)
        x = composer(a, b)
        self.assertEqual(tuple(x.shape), (3, 16))

    def test_injection_is_temporary(self):
        model = DummyModel()
        x = torch.zeros(1, 2, 4)
        delta = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        with inject_last_token(model, layer=0, delta=delta):
            y = model.model.layers[0](x)
            self.assertTrue(torch.equal(y[:, -1, :], delta))

        y2 = model.model.layers[0](x)
        self.assertTrue(torch.equal(y2, x))


if __name__ == "__main__":
    unittest.main()
