import unittest

import torch
from torch import nn

from latent_fork import (
    BilinearComposer,
    LinearComposer,
    final_token_block_trace,
    inject_last_token,
    inject_token_position,
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




class AddBlock(nn.Module):
    def __init__(self, amount):
        super().__init__()
        self.amount = float(amount)

    def forward(self, hidden_states):
        return hidden_states + self.amount


class TraceCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([AddBlock(1.0), AddBlock(2.0), AddBlock(3.0)])


class TraceDummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TraceCore()

    def forward(self, input_ids=None, **_kwargs):
        h = input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
        for block in self.model.layers:
            h = block(h)
        return h


class DummyTokenizer:
    def __call__(self, _text, return_tensors="pt"):
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[1, 2, 3]])}


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


    def test_raw_block_trace_captures_each_decoder_output(self):
        model = TraceDummyModel()
        tok = DummyTokenizer()
        trace = final_token_block_trace(model, tok, "x", device="cpu")
        self.assertEqual(len(trace), 3)
        # Last token starts at value 3 in all four hidden dimensions.
        self.assertTrue(torch.equal(trace[0], torch.full((1, 4), 4.0)))
        self.assertTrue(torch.equal(trace[1], torch.full((1, 4), 6.0)))
        self.assertTrue(torch.equal(trace[2], torch.full((1, 4), 9.0)))

    def test_position_injection_targets_only_requested_token(self):
        model = DummyModel()
        x = torch.zeros(1, 3, 4)
        delta = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        with inject_token_position(model, layer=0, delta=delta, position=1):
            y = model.model.layers[0](x)

        self.assertTrue(torch.equal(y[:, 0, :], torch.zeros(1, 4)))
        self.assertTrue(torch.equal(y[:, 1, :], delta))
        self.assertTrue(torch.equal(y[:, 2, :], torch.zeros(1, 4)))

    def test_once_injection_fires_only_on_first_forward(self):
        model = DummyModel()
        x = torch.zeros(1, 2, 4)
        delta = torch.ones(1, 4)

        with inject_last_token(model, layer=0, delta=delta, once=True):
            y1 = model.model.layers[0](x)
            y2 = model.model.layers[0](x)

        self.assertTrue(torch.equal(y1[:, -1, :], delta))
        self.assertTrue(torch.equal(y2, x))
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
