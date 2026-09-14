"""Complete private checkpoint ensemble loading for reconstructed teacher rounds."""
import gc
from dataclasses import dataclass
from pathlib import Path
import re

from sciona.openvaccine_checkpoints import load_checkpoint
from sciona.openvaccine_ensemble import EXPECTED_MEMBERS, predict_twenty


@dataclass(frozen=True)
class MemberCheckpoint:
    family: str
    slot: int
    directory: Path
    manifest_sha256: str


def predict_teacher(source_dir, checkpoints, nodes, adjacency, *, std_ddof):
    checkpoints=tuple(checkpoints)
    keys=[]
    for record in checkpoints:
        if (not isinstance(record,MemberCheckpoint) or type(record.slot) is not int
                or not isinstance(record.manifest_sha256,str)
                or re.fullmatch('[0-9a-f]{64}',record.manifest_sha256) is None):
            raise ValueError('Invalid teacher checkpoint identity')
        keys.append((record.family,record.slot))
    if len(keys)!=20 or set(keys)!=EXPECTED_MEMBERS:
        raise ValueError('Exactly twenty distinct reconstructed teacher members required')
    def members():
        for record in checkpoints:
            runtime=load_checkpoint(source_dir,record.directory,
                expected_manifest_sha256=record.manifest_sha256,expected_family=record.family)
            yield record.family,record.slot,runtime
            del runtime
            gc.collect()
    return predict_twenty(members(),nodes,adjacency,std_ddof=std_ddof)
