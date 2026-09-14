"""Executable ideal-gas volumetric expansion coefficient."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.physics_ingest.ideal_gas_expansion_proof import SOURCE_VERSION, SOURCE_HASH

PRIMITIVE = 'sciona.atoms.physics.ideal_gas_expansion.ideal_gas_expansion'


def build_ideal_gas_expansion_execution():
    return CDGExport(nodes=[AlgorithmicNode(
        node_id='expansion', name='Ideal-gas volumetric expansion',
        description='Evaluate the reconstructed five-step fixed-pressure derivation alpha=1/T.',
        concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=PRIMITIVE,
        inputs=[IOSpec(name='temperature', type_desc='numpy.ndarray', dim_signature='Th1',
                       constraints='Nonempty finite positive absolute temperature in kelvin; scalar arrays supported. Caller establishes ideal gas at fixed positive pressure and amount.')],
        outputs=[IOSpec(name='volumetric_expansion', type_desc='numpy.ndarray', dim_signature='Th-1',
                        constraints='Positive finite float64 K^-1 with input shape; exact reciprocal rounded once; subnormal outputs accepted.')])],
        edges=[], metadata={'artifact_source':'physics_reconstructed_numerical_realization',
                            'publication_status':'draft','source_version_id':SOURCE_VERSION,
                            'source_content_hash':SOURCE_HASH,'literal_source_parity':False,
                            'num_nodes':1,'num_edges':0,
                            'physical_regime':'Ideal gas PV=nRT with fixed positive pressure and amount in moles; volumetric isobaric coefficient.'})
