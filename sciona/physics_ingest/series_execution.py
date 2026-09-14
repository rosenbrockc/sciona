"""Explicit numerical realization of the reviewed-source series derivation.

This creates an execution graph for local validation, not a publication decision.
The source derivation and all dependency/review requirements remain separate.
"""
from uuid import UUID
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,IOSpec,NodeStatus
from sciona.ghost.dimensions import DimensionalSignature

PRIMITIVE='sciona.atoms.electrical.series_resistance.series_resistance'


def build_series_execution_cdg(*,source_version_id:str,runtime_evidence_id:str):
    source_version_id=str(UUID(source_version_id));runtime_evidence_id=str(UUID(runtime_evidence_id))
    resistance=DimensionalSignature(M=1,L=2,T=-3,I=-2).to_compact()
    current=DimensionalSignature(I=1).to_compact()
    return CDGExport(nodes=[AlgorithmicNode(
        node_id='series_equivalent_resistance',name='Evaluate two-component series resistance',
        description='Ohmic series model with explicit nonzero-current derivation guard.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PRIMITIVE,
        inputs=[
            IOSpec(name='resistance_a',type_desc='numpy.ndarray',constraints='finite real, nonnegative ohms; NumPy broadcasting',dim_signature=resistance),
            IOSpec(name='resistance_b',type_desc='numpy.ndarray',constraints='finite real, nonnegative ohms; NumPy broadcasting',dim_signature=resistance),
            IOSpec(name='current',type_desc='numpy.ndarray',constraints='finite real, nonzero amperes; NumPy broadcasting',dim_signature=current),
        ],
        outputs=[IOSpec(name='equivalent_resistance',type_desc='numpy.ndarray',dim_signature=resistance)],
    )],edges=[],metadata={
        'artifact_source':'physics_numerical_realization', 'publication_status':'draft',
        'source_derivation_version_id':source_version_id,
        'validated_terminal_runtime_evidence_id':runtime_evidence_id,
        'requires_source_dependency_review':True,
        'physical_regime':'lumped ohmic components in a single series branch',
        'num_nodes':1,'num_edges':0,
    })
