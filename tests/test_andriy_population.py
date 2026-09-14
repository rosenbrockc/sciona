import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.andriy_population import andriy_documented_population


@pytest.mark.parametrize('kind',['sequence','groups','rate','duration','channels','nonfinite'])
def test_raw_population_rejects_invalid_contract_before_processing(kind):
    x=np.zeros((16,76800))
    candidates=[x]*4
    seq=np.array([1,6,1,6])
    p,aux,n,t=[x],[],[x],[x]
    fs=128
    if kind=='sequence': seq[0]=0
    elif kind=='groups': n=[]
    elif kind=='rate': fs=True
    elif kind=='duration': t=[x[:,:-1]]
    elif kind=='channels': p=[x[:-1]]
    else:
        x=x.copy();x[0,0]=np.nan;t=[x]
    with pytest.raises(ValueError):
        andriy_documented_population(candidates,seq,p,aux,n,t,fs)
