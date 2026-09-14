"""Synthesis-time harmonic wave to Helmholtz realization."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,IOSpec,NodeStatus
from sciona.physics_ingest.helmholtz_proof import SOURCE_VERSION,SOURCE_HASH

PRIMITIVE='sciona.atoms.physics.helmholtz.helmholtz'


def build_helmholtz_execution():
    inputs=[IOSpec(name=n,type_desc='str',constraints=c) for n,c in [
        ('amplitude_srepr','Tuple of three time-independent C2 spatial amplitudes with consistent caller-established field units.'),
        ('coordinates_srepr','Tuple of distinct real Cartesian Symbols x,y,z,t.'),
        ('angular_frequency_srepr','Real constant omega, coordinate independent; zero and negative allowed.'),
        ('wave_speed_srepr','Positive constant c, coordinate independent; consistent speed units.')]]
    outputs=[IOSpec(name=n,type_desc='str',constraints=c) for n,c in [
        ('harmonic_field_srepr','Tuple U exp(+i omega t).'),
        ('laplacian_srepr','Tuple of componentwise formal spatial Laplacians.'),
        ('helmholtz_residual_srepr','Tuple Laplacian(U)+(omega/c)^2 U; nonzero residuals retained.')]]
    return CDGExport(nodes=[AlgorithmicNode(node_id='harmonic',name='Reduce harmonic wave field to Helmholtz residual',
        description='Construct time-harmonic field and full componentwise Helmholtz residual, without assuming a solution.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PRIMITIVE,inputs=inputs,outputs=outputs)],edges=[],
        metadata={'artifact_source':'physics_reconstructed_symbolic_realization','publication_status':'draft',
                  'source_version_id':SOURCE_VERSION,'source_content_hash':SOURCE_HASH,'literal_source_parity':False,
                  'num_nodes':1,'num_edges':0,'execution_phase':'symbolic synthesis',
                  'physical_regime':'Time-independent C2 Cartesian amplitude, real constant frequency and positive constant speed. No Maxwell divergence certification; caller establishes units and domain.'})
