"""Connected s32 training numerics with externally supplied stochastic masks.

MIT model adaptation; see docs/licenses/WebTraffic-MIT.txt. No optimizer or checkpoint IO.
"""
from .webtraffic_encoder import encode
from .webtraffic_dropout import apply_mask, decode_with_masks
from .webtraffic_losses import calc_loss, decode_predictions, rnn_activation_loss


def s32_forward(x_features, y_features, previous_y, mean, std, true_y,
                encoder_cells, decoder_cells, projection_kernel, projection_bias,
                gate_mask=None, decoder_masks=None, training=True):
    """Source one-layer, width267, 283-to-63-day route; explicit masks during training."""
    if len(encoder_cells) != 1 or len(decoder_cells) != 1:
        raise ValueError('s32 requires one encoder and one decoder layer')
    if x_features.shape[1] != 283 or y_features.shape[1] != 63:
        raise ValueError('s32 final route requires 283 input and 63 forecast days')
    if encoder_cells[0]['candidate_hidden_bias'].shape != (267,):
        raise ValueError('s32 encoder width must be 267')
    if decoder_cells[0]['candidate_bias'].shape != (267,):
        raise ValueError('s32 decoder width must be 267')
    encoder_output, state = encode(x_features,encoder_cells)
    # cuDNN dropout is between layers; this source configuration has one layer.
    initial = apply_mask(state[0],gate_mask,.9967589439360334 if training else 1.)
    keep = dict(input=1.,state=.99,output=.975) if training else dict(input=1.,state=1.,output=1.)
    if training and decoder_masks is None:
        raise ValueError('Explicit decoder training masks required')
    masks = decoder_masks if training else [{}]
    targets, decoder_output, final_state = decode_with_masks(
        (initial,),y_features,previous_y,decoder_cells,projection_kernel,projection_bias,masks,[keep])
    prediction = decode_predictions(targets,mean,std)
    mae,smooth,rounded,count = calc_loss(prediction,true_y)
    encoder_penalty = rnn_activation_loss(encoder_output,1e-6/283)
    decoder_penalty = rnn_activation_loss(decoder_output,5e-6/63)
    return dict(predictions=prediction,mae=mae,smooth=smooth,rounded=rounded,item_count=count,
                encoder_penalty=encoder_penalty,decoder_penalty=decoder_penalty,
                total_loss=smooth+encoder_penalty+decoder_penalty,
                final_state=final_state,encoder_output=encoder_output,decoder_output=decoder_output)
