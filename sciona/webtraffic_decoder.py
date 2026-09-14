"""Web Traffic decoder arithmetic with explicit runtime weights.

Model loop adapted under MIT; see docs/licenses/WebTraffic-MIT.txt.
Cell equations follow TensorFlow 1.10 GRUBlockCell's documented mathematics.
No TensorFlow initialization, dropout RNG or checkpoint-codec equivalence claimed.
"""
import torch


def gru_block_step(x, state, gate_kernel, gate_bias, candidate_kernel, candidate_bias):
    gates = torch.sigmoid(torch.cat((x, state), dim=1) @ gate_kernel + gate_bias)
    reset, update = gates.chunk(2, dim=1)
    candidate = torch.tanh(torch.cat((x, state * reset), dim=1) @ candidate_kernel + candidate_bias)
    return (1 - update) * candidate + update * state


def convert_encoder_state(state, decoder_layers):
    """Source state mapping without dropout: upper layers or zero-padded top layers."""
    layers = list(state.unbind(0))
    if len(layers) >= decoder_layers:
        return tuple(layers[len(layers)-decoder_layers:])
    return tuple(layers + [torch.zeros_like(layers[0]) for _ in range(decoder_layers-len(layers))])


def decode(encoder_state, features, previous_y, cells, projection_kernel,
           projection_bias, attention=None):
    """Autoregressive source recurrence without dropout; returns time-first arrays."""
    states = tuple(encoder_state)
    if len(states) != len(cells) or not cells:
        raise ValueError('One state per decoder cell required')
    previous = previous_y[:,None]
    targets, outputs = [], []
    for day in range(features.shape[1]):
        parts = [previous, features[:,day]]
        if attention is not None:
            parts.append(attention[:,day])
        value = torch.cat(parts,dim=1)
        updated = []
        for state, weights in zip(states, cells):
            value = gru_block_step(value, state, **weights)
            updated.append(value)
        states = tuple(updated)
        previous = value @ projection_kernel + projection_bias
        outputs.append(value)
        targets.append(previous.squeeze(-1))
    return torch.stack(targets), torch.stack(outputs), states
