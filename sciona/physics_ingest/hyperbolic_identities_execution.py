"""Complete real-argument hyperbolic identity collection with symbolic certificate."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.physics_ingest.hyperbolic_identities_proof import SOURCE_VERSION, SOURCE_HASH
PRIMITIVE = 'sciona.atoms.physics.hyperbolic_identities.hyperbolic_identities'


def build_hyperbolic_identities_execution():
    return CDGExport(nodes=[AlgorithmicNode(node_id='identities', name='Hyperbolic identities',
        description='Return six symbolic function forms, eighteen specialized identity pairs and an exponential-definition certificate.',
        concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=PRIMITIVE,
        inputs=[IOSpec(name='argument_srepr', type_desc='str', dim_signature='1',
                       constraints='Guarded finite real commutative dimensionless scalar srepr; caller establishes units.')],
        outputs=[IOSpec(name=n, type_desc='str', dim_signature='1', constraints=c) for n, c in [
            ('functions_json', 'JSON mapping of six function srepr forms at the supplied argument.'),
            ('identities_json', 'JSON list of eighteen specialized identity pairs, with one-based step indices.'),
            ('certificate_json', 'Self-contained generic exponential-definition proof and bound specialized outputs.')]])], edges=[],
        metadata={'artifact_source':'physics_reconstructed_symbolic_realization', 'publication_status':'draft',
                  'source_version_id':SOURCE_VERSION, 'source_content_hash':SOURCE_HASH, 'literal_source_parity':False,
                  'num_nodes':1, 'num_edges':0, 'mathematical_regime':'Finite real dimensionless argument; positive hyperbolic denominator, complex sine/cosine definitions for imaginary arguments only. No numeric cancellation claim.'})
