"""Executable ideal-gas isothermal compressibility."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.physics_ingest.ideal_gas_compressibility_proof import SOURCE_VERSION, SOURCE_HASH

PRIMITIVE = 'sciona.atoms.physics.ideal_gas_compressibility.ideal_gas_compressibility'


def build_ideal_gas_compressibility_execution():
    return CDGExport(nodes=[AlgorithmicNode(
        node_id='compressibility', name='Ideal-gas isothermal compressibility',
        description='Evaluate the reconstructed six-step fixed-temperature derivation kappa=1/P.',
        concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=PRIMITIVE,
        inputs=[IOSpec(name='pressure', type_desc='numpy.ndarray', dim_signature='M1 L-1 T-2',
                       constraints='Nonempty finite positive absolute pressure in pascals; scalar arrays supported. Caller establishes ideal gas at fixed positive temperature and amount.')],
        outputs=[IOSpec(name='isothermal_compressibility', type_desc='numpy.ndarray', dim_signature='M-1 L1 T2',
                        constraints='Positive finite float64 Pa^-1 with input shape; exact reciprocal rounded once; subnormal outputs accepted.')])],
        edges=[], metadata={'artifact_source':'physics_reconstructed_numerical_realization',
                            'publication_status':'draft','source_version_id':SOURCE_VERSION,
                            'source_content_hash':SOURCE_HASH,'literal_source_parity':False,
                            'num_nodes':1,'num_edges':0,
                            'physical_regime':'Ideal gas PV=nRT with fixed positive temperature and amount in moles; isothermal volume compressibility.'})
