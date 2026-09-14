"""Executable periodic-wave parameter relations."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.physics_ingest.circular_orbit_speed_proof import SOURCE_VERSION, SOURCE_HASH

PRIMITIVE = 'sciona.atoms.physics.circular_orbit_speed.circular_orbit_speed'


def build_circular_orbit_speed_execution():
    inputs=[IOSpec(name=n,type_desc='numpy.ndarray',dim_signature=d,constraints=c) for n,d,c in [
        ('radius','L1','Positive finite circular-path radius in meters; identical nonempty input shapes, no broadcasting.'),
        ('period','T1','Positive finite circuit period in seconds; caller supplies explicit calendar conversion if needed.')]]
    outputs=[IOSpec(name=n,type_desc='numpy.ndarray',dim_signature=d,constraints=c) for n,d,c in [
        ('circumference','L1','Positive finite circumference in meters, independently rounded; input shape.'),
        ('average_speed','L1 T-1','Positive finite circuit-average speed in m/s; instantaneous only for uniform circular motion.')]]
    return CDGExport(nodes=[AlgorithmicNode(node_id='orbit',name='Circular-path average speed',
        description='Compute circumference and average speed using radius and period without premature rounding.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PRIMITIVE,inputs=inputs,outputs=outputs)],edges=[],
        metadata={'artifact_source':'physics_reconstructed_numerical_realization','publication_status':'draft',
                  'source_version_id':SOURCE_VERSION,'source_content_hash':SOURCE_HASH,'literal_source_parity':False,
                  'num_nodes':1,'num_edges':0,'physical_regime':'Circular-path average speed; uniform motion required for instantaneous interpretation. No ephemeris or gravitational dynamics claim.'})
