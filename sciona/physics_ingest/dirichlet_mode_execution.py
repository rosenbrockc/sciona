"""Normalized interval mode with both sign representatives and certificate."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.physics_ingest.dirichlet_mode_proof import SOURCE_VERSION, SOURCE_HASH
PRIMITIVE = 'sciona.atoms.physics.dirichlet_mode.dirichlet_mode'


def build_dirichlet_mode_execution():
    inputs = [IOSpec(name=n, type_desc='str', dim_signature=d, constraints=c) for n,d,c in [
        ('width_srepr','L','Guarded srepr of positive finite real interval width, constant with respect to x.'),
        ('mode_index_srepr','1','Guarded srepr of positive integer mode index, including general symbolic integers.'),
        ('phase_srepr','1','Guarded srepr of finite real constant global phase; zero and pi recover source signs.')]]
    outputs = [IOSpec(name=n, type_desc='str', dim_signature=d, constraints=c) for n,d,c in [
        ('wave_srepr','L^-1/2','Normalized mode on [0,width], fixed real x.'),
        ('opposite_wave_srepr','L^-1/2','Opposite global sign, same eigenvalue and normalized density.'),
        ('wavenumber_srepr','L^-1','Positive n*pi/width.'),
        ('derivative_srepr','L^-3/2','First spatial derivative in interval interior.'),
        ('second_derivative_srepr','L^-5/2','Second spatial derivative in interval interior.'),
        ('certificate_json','1','Metadata with input ASTs, eigenvalue, density antiderivative and seven checked identities.')]]
    return CDGExport(nodes=[AlgorithmicNode(node_id='mode',name='Normalized Dirichlet interval mode',
        description='Construct both global-sign representatives, wavenumber, derivatives and normalization certificate.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PRIMITIVE,inputs=inputs,outputs=outputs)],edges=[],metadata={
        'artifact_source':'physics_reconstructed_symbolic_realization','publication_status':'draft',
        'source_version_id':SOURCE_VERSION,'source_content_hash':SOURCE_HASH,'literal_source_parity':False,
        'num_nodes':1,'num_edges':0,
        'physical_regime':'Positive interval width and positive integer mode index, constant real phase, fixed real x. Interior -psi_second=k^2*psi with endpoint Dirichlet values and unit L2 norm. No exterior derivative, finite-well, energy-conversion or completeness claim.'})
