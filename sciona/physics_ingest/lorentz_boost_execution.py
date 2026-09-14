"""Standard x-axis Lorentz boost on event coordinates with explicit c and v."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,IOSpec,NodeStatus
from sciona.physics_ingest.lorentz_boost_proof import SOURCE_VERSION,SOURCE_HASH
PRIMITIVE='sciona.atoms.physics.lorentz_boost.lorentz_boost'


def build_lorentz_boost_execution():
    inputs=[IOSpec(name=n,type_desc='numpy.ndarray',dim_signature=d,constraints=c) for n,d,c in [
        ('light_speed','L T^-1','Positive finite c; identical nonempty input shapes.'),
        ('relative_velocity','L T^-1','Signed frame velocity along x with |v|<c; zero allowed.'),
        ('time','T','Finite signed event time; common origin at t=t_prime=0.'),
        ('x','L','Finite signed x coordinate in original frame.'),
        ('y','L','Finite signed transverse y coordinate.'),
        ('z','L','Finite signed transverse z coordinate.')]]
    outputs=[IOSpec(name=n,type_desc='numpy.ndarray',dim_signature=d,constraints=c) for n,d,c in [
        ('gamma','1','Positive Lorentz factor, independently rounded.'),
        ('time_prime','T','gamma*(t-v*x/c²), independently rounded.'),
        ('x_prime','L','gamma*(x-v*t), independently rounded.'),
        ('y_prime','L','Unchanged transverse coordinate, independent output array.'),
        ('z_prime','L','Unchanged transverse coordinate, independent output array.')]]
    return CDGExport(nodes=[AlgorithmicNode(node_id='boost',name='Standard Lorentz boost',
        description='Transform event coordinates to an inertial frame moving at signed subluminal velocity along x.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PRIMITIVE,inputs=inputs,outputs=outputs)],edges=[],
        metadata={'artifact_source':'physics_reconstructed_numerical_realization','publication_status':'draft',
                  'source_version_id':SOURCE_VERSION,'source_content_hash':SOURCE_HASH,'literal_source_parity':False,
                  'num_nodes':1,'num_edges':0,'physical_regime':'Standard inertial x boost, common origin and aligned axes, c>0 and |v|<c. Positive gamma and identity at v=0. No exact invariant or inverse guarantee after float64 output rounding.'})
