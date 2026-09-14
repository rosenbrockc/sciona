"""Executable corrected ideal-projectile time elimination."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.physics_ingest.projectile_proof import SOURCE_VERSION, SOURCE_HASH

PRIMITIVE = 'sciona.atoms.physics.projectile_trajectory.projectile_trajectory'


def build_projectile_execution():
    ports = []
    for name, dimension, units in [('x','L1','m'),('x0','L1','m'),('y0','L1','m'),
                                   ('vx0','L1 T-1','m/s'),('vy0','L1 T-1','m/s'),('gravity','L1 T-2','m/s^2')]:
        ports.append(IOSpec(name=name, type_desc='numpy.ndarray', dim_signature=dimension,
            constraints='Finite real '+units+'; identical nonempty shape for all ports, no broadcasting. vx0!=0, gravity>=0, (x-x0)/vx0>=0.'))
    return CDGExport(nodes=[AlgorithmicNode(node_id='trajectory', name='Evaluate corrected projectile trajectory',
        description='Eliminate time from constant-acceleration kinematics with initial vertical velocity and linear gravity.',
        concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=PRIMITIVE, inputs=ports,
        outputs=[IOSpec(name='elapsed_time', type_desc='numpy.ndarray', dim_signature='T1', constraints='Nonnegative finite float64 seconds; input shape preserved.'),
                 IOSpec(name='height', type_desc='numpy.ndarray', dim_signature='L1', constraints='Finite float64 meters, positive upward; exact time used before independent output rounding.')])], edges=[],
        metadata={'artifact_source':'physics_reconstructed_numerical_realization','publication_status':'draft',
                  'source_version_id':SOURCE_VERSION,'source_content_hash':SOURCE_HASH,'literal_source_parity':False,
                  'num_nodes':1,'num_edges':0,'physical_regime':'Ideal no-drag, constant downward acceleration, no collision cutoff; forward time and nonzero horizontal velocity.'})
