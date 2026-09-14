"""Synthesis-time symbolic integration-by-parts realization."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,IOSpec,NodeStatus
from sciona.physics_ingest.integration_parts_proof import SOURCE_VERSION,SOURCE_HASH

PRIMITIVE='sciona.atoms.physics.integration_by_parts.integration_by_parts'


def build_integration_parts_execution():
    inputs=[IOSpec(name=n,type_desc='str',constraints=c) for n,c in [
        ('u_srepr','Reviewed scalar commutative srepr AST for u; C1 on a common connected real interval.'),
        ('v_srepr','Reviewed scalar commutative srepr AST for v; C1 on the same interval.'),
        ('variable_srepr','srepr AST of a real Symbol; consistent symbol assumptions across all inputs.'),
        ('constant_srepr','Finite scalar commutative srepr AST independent of the integration variable.')]]
    return CDGExport(nodes=[AlgorithmicNode(node_id='parts',name='Rewrite integral by parts',
        description='Parametrized differential identity with explicit constant and unevaluated residual integral; derivative verified.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PRIMITIVE,inputs=inputs,
        outputs=[IOSpec(name='integrand_srepr',type_desc='str',constraints='Portable AST of u*Derivative(v,x).'),
                 IOSpec(name='antiderivative_srepr',type_desc='str',constraints='Portable AST of u*v-Integral(v*Derivative(u,x),x)+C. Residual integral retained.')])],edges=[],
        metadata={'artifact_source':'physics_reconstructed_symbolic_realization','publication_status':'draft',
                  'source_version_id':SOURCE_VERSION,'source_content_hash':SOURCE_HASH,'literal_source_parity':False,
                  'num_nodes':1,'num_edges':0,'execution_phase':'symbolic synthesis',
                  'physical_regime':'C1 scalar functions of a real parameter on a caller-established connected interval; indefinite antiderivative modulo independent constant.'})
