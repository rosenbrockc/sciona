"""Synthesis-time Cartesian curl-curl realization."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,IOSpec,NodeStatus
from sciona.physics_ingest.curl_curl_proof import SOURCE_VERSION,SOURCE_HASH

PRIMITIVE='sciona.atoms.physics.curl_curl.curl_curl'


def build_curl_curl_execution():
    inputs=[IOSpec(name=n,type_desc='str',constraints=c) for n,c in [
        ('field_srepr','Tuple of three finite scalar commutative C2 field components on a common open Cartesian region; consistent physical units.'),
        ('coordinates_srepr','Tuple of three distinct real Symbols in a fixed right-handed orthonormal Cartesian frame.')]]
    outputs=[IOSpec(name=n,type_desc='str',constraints=c) for n,c in [
        ('curl_curl_srepr','Tuple of formal curl-curl components.'),
        ('gradient_divergence_srepr','Tuple of formal gradient-of-divergence components; retained without divergence-free assumption.'),
        ('laplacian_srepr','Tuple of componentwise Laplacians. First output equals second minus third under C2 Cartesian assumptions.')]]
    return CDGExport(nodes=[AlgorithmicNode(node_id='curl_identity',name='Construct Cartesian curl-curl identity',
        description='General symbolic vector identity from corrected Levi-Civita contraction, retaining divergence term.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PRIMITIVE,inputs=inputs,outputs=outputs)],edges=[],
        metadata={'artifact_source':'physics_reconstructed_symbolic_realization','publication_status':'draft',
                  'source_version_id':SOURCE_VERSION,'source_content_hash':SOURCE_HASH,'literal_source_parity':False,
                  'num_nodes':1,'num_edges':0,'execution_phase':'symbolic synthesis',
                  'physical_regime':'C2 Cartesian vector fields; no Maxwell or divergence-free assumption. Caller establishes units, smoothness and domain.'})
