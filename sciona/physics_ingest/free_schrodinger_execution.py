"""Full symbolic free-particle mode and its differential certificate."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.physics_ingest.free_schrodinger_proof import SOURCE_VERSION, SOURCE_HASH

PRIMITIVE = 'sciona.atoms.physics.free_schrodinger.free_schrodinger'


def build_free_schrodinger_execution():
    inputs = [IOSpec(name=n, type_desc='str', dim_signature=d, constraints=c) for n, d, c in [
        ('mass_srepr', 'M', 'Guarded srepr of provably positive finite real constant mass.'),
        ('hbar_srepr', 'L^2 M T^-1', 'Guarded srepr of provably positive finite real reduced Planck constant.'),
        ('momentum_srepr', 'L M T^-1', 'Guarded srepr Tuple of three finite real constant Cartesian momentum components.'),
        ('amplitude_srepr', 'L^-3/2', 'Guarded srepr of finite complex constant amplitude; zero allowed, no normalization claim.')]]
    outputs = [IOSpec(name=n, type_desc='str', dim_signature=d, constraints=c) for n, d, c in [
        ('wave_srepr', 'L^-3/2', 'Symbolic wavefunction on fixed real x,y,z,t; exp(+i*(p dot r-E*t)/hbar).'),
        ('energy_srepr', 'L^2 M T^-2', 'Exact kinetic energy p dot p/(2*m).'),
        ('gradient_json', 'L^-5/2', 'JSON list of three Cartesian derivative srepr strings.'),
        ('laplacian_srepr', 'L^-7/2', 'Symbolic Cartesian Laplacian, -p dot p*wave/hbar^2.'),
        ('time_derivative_srepr', 'L^-3/2 T^-1', 'Symbolic time derivative -i*E*wave/hbar.'),
        ('certificate_json', '1', 'Parameter and coordinate ASTs, Hamiltonian action and seven checked identities; metadata, not a physical scalar.')]]
    return CDGExport(nodes=[AlgorithmicNode(node_id='wave', name='Free Schrödinger plane wave',
        description='Construct a three-dimensional symbolic plane wave and verify its spatial, temporal and Hamiltonian identities.',
        concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=PRIMITIVE,
        inputs=inputs, outputs=outputs)], edges=[], metadata={
        'artifact_source': 'physics_reconstructed_symbolic_realization', 'publication_status': 'draft',
        'source_version_id': SOURCE_VERSION, 'source_content_hash': SOURCE_HASH, 'literal_source_parity': False,
        'num_nodes': 1, 'num_edges': 0,
        'physical_regime': 'Three-dimensional free nonrelativistic scalar mode, U=0, constant parameters; positive mass/hbar, real momentum and finite complex amplitude. Zero momentum/amplitude allowed. Fixed real x,y,z,t; no normalized-state, boundary, potential, relativistic or arbitrary-solution claim.'})
