"""cuDNN-compatible GRU equations with explicit canonical runtime weights.

Mathematics documented by TensorFlow 1.10 CudnnCompatibleGRUCell.
No opaque cuDNN checkpoint or seeded initializer compatibility is claimed.
"""
import torch
from .webtraffic_decoder import convert_encoder_state, decode
from .webtraffic_losses import decode_predictions


def encode(features, cells):
    """Batch-first inputs, time-first outputs and layer-first final states; no dropout."""
    values = features.transpose(0, 1)
    states = []
    for weights in cells:
        depth = weights['candidate_hidden_bias'].shape[0]
        state = values.new_zeros((features.shape[0], depth))
        outputs = []
        for x in values.unbind(0):
            gates = torch.sigmoid(torch.cat([x, state], dim=1) @ weights['gate_kernel'] + weights['gate_bias'])
            reset, update = gates.chunk(2, dim=1)
            candidate = torch.tanh(x @ weights['candidate_input_kernel'] + weights['candidate_input_bias'] +
                                   reset * (state @ weights['candidate_hidden_kernel'] + weights['candidate_hidden_bias']))
            state = (1 - update) * candidate + update * state
            outputs.append(state)
        values = torch.stack(outputs)
        states.append(state)
    return values, torch.stack(states)


def predict_without_dropout(x_features, y_features, previous_y, mean, std,
                            encoder_cells, decoder_cells, projection_kernel, projection_bias):
    """Connected source s32 prediction path, whose decoder attention is disabled."""
    encoder_output, state = encode(x_features, encoder_cells)
    mapped = convert_encoder_state(state, len(decoder_cells))
    targets, decoder_output, final_state = decode(mapped, y_features, previous_y,
                                                 decoder_cells, projection_kernel, projection_bias)
    predictions = decode_predictions(targets, mean, std)
    return predictions, encoder_output, decoder_output, final_state
