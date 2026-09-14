"""Private folding via the reviewed CONTRAfold-SE/EternaFold build."""
import hashlib
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from sciona.openvaccine_features import folded_features
from sciona.openvaccine_structure import loop_labels

BINARY_SHA256='892a04952375e6ed540bb7695852108e80e90a39c9cbc1e4ae11d6b3fe04eaef'
PARAMETERS_SHA256='1421e89cc8df24b53a320eff6c72b8acfdb2771faba29ab9c798f7bdfc853a8a'


def fold_sequence(sequence,*,binary,parameters,timeout=30):
    if not isinstance(sequence,str) or len(sequence)<2 or set(sequence)-set('ACGU'):
        raise ValueError('Supported RNA sequence of length at least two required')
    if isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not np.isfinite(timeout) or timeout<=0:
        raise ValueError('Finite positive folding timeout required')
    binary,parameters=Path(binary).resolve(),Path(parameters).resolve()
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=BINARY_SHA256:
        raise ValueError('Folding executable identity mismatch')
    if hashlib.sha256(parameters.read_bytes()).hexdigest()!=PARAMETERS_SHA256:
        raise ValueError('Folding parameter identity mismatch')
    with tempfile.TemporaryDirectory(prefix='sciona_fold_') as temp:
        root=Path(temp);input_path=root/'input';posterior_path=root/'posterior'
        input_path.write_text(''.join(f'{i+1} {base} -1\n' for i,base in enumerate(sequence)))
        command=[str(binary),'predict',str(input_path),'--params',str(parameters)]
        def run(extra):
            try:
                result=subprocess.run(command+extra,cwd=root,capture_output=True,text=True,timeout=timeout)
            except (subprocess.TimeoutExpired,OSError):
                raise RuntimeError('Folding subprocess failed or timed out') from None
            if result.returncode!=0:
                # The engine's stderr can contain private input. Do not expose it.
                raise RuntimeError('Folding subprocess returned an error')
            return result.stdout
        output=run([])
        lines=output.strip().splitlines()
        structure=lines[-1] if lines else ''
        if len(structure)!=len(sequence):raise ValueError('Invalid folded structure length')
        loop_labels(structure)
        run(['--posteriors','0.0000000001',str(posterior_path)])
        matrix=np.zeros((len(sequence),len(sequence)),dtype=np.float64)
        try:
            rows=posterior_path.read_text().splitlines()
            if len(rows)!=len(sequence):raise ValueError()
            for expected,line in enumerate(rows,start=1):
                fields=line.split()
                if len(fields)<2 or int(fields[0])!=expected or fields[1]!=sequence[expected-1]:raise ValueError()
                seen=set()
                for field in fields[2:]:
                    index,value=field.split(':');j=int(index)-1;p=float(value)
                    if j<=expected-1 or j>=len(sequence) or j in seen or not np.isfinite(p) or not 0<=p<=1:raise ValueError()
                    seen.add(j);matrix[expected-1,j]=matrix[j,expected-1]=p
        except (ValueError,OSError,IndexError):
            raise ValueError('Invalid folding posterior output') from None
        if (matrix.sum(axis=1)>1+1e-6).any():raise ValueError('Invalid folding posterior normalization')
        return structure,matrix


def fold_features(source_dir,sequences,*,binary,parameters,timeout=30):
    if not isinstance(sequences,(tuple,list)) or not sequences or any(not isinstance(s,str) for s in sequences):
        raise ValueError('Explicit sequence population required')
    if len({len(s) for s in sequences})!=1:raise ValueError('Equal-length feature batch required')
    folded=[fold_sequence(s,binary=binary,parameters=parameters,timeout=timeout) for s in sequences]
    return folded_features(source_dir,sequences,[v[0] for v in folded],[v[1] for v in folded])
