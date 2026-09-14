"""Exact finite-matrix execution of the reconstructed eigenstate identity."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,IOSpec,NodeStatus
from sciona.physics_ingest.eigenstate_orthogonality_proof import SOURCE_VERSION,SOURCE_HASH

PRIMITIVE='sciona.atoms.physics.eigenstate_orthogonality.eigenstate_orthogonality'
INPUT_NAMES=['operator_srepr','initial_state_srepr','final_state_srepr','initial_eigenvalue_srepr','final_eigenvalue_srepr']
OUTPUT_NAMES=['overlap_srepr','matrix_element_srepr','gap_overlap_srepr']


def build_eigenstate_orthogonality_execution():
    constraints=['Nonempty square nested Tuple of finite scalar expressions; exactly Hermitian.',
                 'Nonzero Tuple eigenstate u in an orthonormal finite basis; dimensionless components.',
                 'Matching nonzero Tuple eigenstate v; no normalization requirement.',
                 'Finite real eigenvalue a satisfying Au=au exactly; common operator units.',
                 'Finite real eigenvalue b satisfying Av=bv exactly; may equal a.']
    return CDGExport(nodes=[AlgorithmicNode(node_id='eigenstates',name='Calculate Hermitian eigenstate overlap identity',
        description='Verify Hermiticity and eigenstates, then calculate complex overlap, matrix element and eigenvalue-gap product.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PRIMITIVE,
        inputs=[IOSpec(name=n,type_desc='str',constraints=c) for n,c in zip(INPUT_NAMES,constraints)],
        outputs=[IOSpec(name=n,type_desc='str',constraints=c) for n,c in zip(OUTPUT_NAMES,
            ['Exact u†v; may be nonzero for degenerate eigenvalues.','Exact u†Av in operator units.',
             'Computed (b-a)u†v; zero under verified premises.'])])],edges=[],
        metadata={'artifact_source':'physics_reconstructed_symbolic_realization','publication_status':'draft',
                  'source_version_id':SOURCE_VERSION,'source_content_hash':SOURCE_HASH,'literal_source_parity':False,
                  'num_nodes':1,'num_edges':0,'execution_phase':'symbolic synthesis',
                  'physical_regime':'Finite Hermitian operator in orthonormal basis; exact eigenstates; degenerate states allowed. Caller establishes consistent operator units.'})
