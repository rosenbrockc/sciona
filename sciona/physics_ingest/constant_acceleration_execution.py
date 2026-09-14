"""Signed constant-acceleration motion, including zero time and reversal."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,IOSpec,NodeStatus
from sciona.physics_ingest.constant_acceleration_proof import SOURCE_VERSION,SOURCE_HASH
PRIMITIVE='sciona.atoms.physics.constant_acceleration.constant_acceleration'


def build_constant_acceleration_execution():
    inputs=[IOSpec(name=n,type_desc='numpy.ndarray',dim_signature=d,constraints=c) for n,d,c in [
        ('initial_velocity','L T^-1','Finite signed one-dimensional initial velocity; identical nonempty shapes.'),
        ('acceleration','L T^-2','Finite signed constant acceleration, including zero.'),
        ('elapsed_time','T','Finite nonnegative elapsed time, including zero.')]]
    outputs=[IOSpec(name=n,type_desc='numpy.ndarray',dim_signature=d,constraints=c) for n,d,c in [
        ('final_velocity','L T^-1','Signed final velocity from u+a*t, not a square-root sign guess.'),
        ('displacement','L','Signed displacement; not total distance during reversal.'),
        ('average_velocity','L T^-1','Time average for positive duration; continuous extension u at t=0.')]]
    return CDGExport(nodes=[AlgorithmicNode(node_id='motion',name='Constant-acceleration motion',
        description='Evaluate signed final velocity, displacement and mean velocity with exact intermediate arithmetic.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PRIMITIVE,inputs=inputs,outputs=outputs)],edges=[],
        metadata={'artifact_source':'physics_reconstructed_numerical_realization','publication_status':'draft',
                  'source_version_id':SOURCE_VERSION,'source_content_hash':SOURCE_HASH,'literal_source_parity':False,
                  'num_nodes':1,'num_edges':0,'physical_regime':'One-dimensional constant acceleration; signed velocities and displacement, nonnegative elapsed time. Zero-time average is a continuous extension; no path-length or inverse-sign claim.'})
