"""Source-ordered manual/prediction assembly for Amex tree and neural learners."""
import numpy as np
from sciona.amex_numeric import _matrix
from sciona.amex_binning import normalize
from sciona.amex_sequence import prediction_slots

BLOCK_ORDER=('full_categorical','full_numeric','full_difference','customer_rank','last3_categorical','last3_numeric','last3_difference','last6_numeric','month_rank')


def assemble(blocks,row_predictions):
    if type(blocks) is not dict or set(blocks)!=set(BLOCK_ORDER):raise ValueError('All nine manual feature blocks required')
    arrays=[_matrix(blocks[name]) for name in BLOCK_ORDER]
    if any(len(a)!=len(arrays[0]) for a in arrays):raise ValueError('Aligned manual populations required')
    slots=prediction_slots(row_predictions)
    if len(slots)!=len(arrays[0]):raise ValueError('Aligned row prediction populations required')
    tree=np.concatenate([*arrays,slots],axis=1)
    neural=np.concatenate([*[normalize(a)['values'] for a in arrays],np.nan_to_num(slots,nan=0.)],axis=1)
    return dict(tree_features=tree,neural_features=neural,block_widths=[a.shape[1] for a in arrays],prediction_width=13)
