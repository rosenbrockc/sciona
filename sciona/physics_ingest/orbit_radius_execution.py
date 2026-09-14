"""Circular test-mass orbit from explicit G, central mass and period."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,IOSpec,NodeStatus
from sciona.physics_ingest.orbit_radius_proof import SOURCE_VERSION,SOURCE_HASH
PRIMITIVE='sciona.atoms.physics.orbit_radius.orbit_radius'


def build_orbit_radius_execution():
    inputs=[IOSpec(name=n,type_desc='numpy.ndarray',dim_signature=d,constraints=c) for n,d,c in [
        ('gravitational_constant','L^3 M^-1 T^-2','Positive finite G in SI; identical nonempty input shapes.'),
        ('central_mass','M','Positive finite central mass; negligible satellite test mass.'),
        ('period','T','Positive finite circular orbital period in seconds; no automatic geostationary certification.')]]
    outputs=[IOSpec(name=n,type_desc='numpy.ndarray',dim_signature=d,constraints=c) for n,d,c in [
        ('radius','L','Positive radius measured from central body center, not surface altitude.'),
        ('orbital_speed','L T^-1','Positive uniform circular speed, independently rounded.')]]
    return CDGExport(nodes=[AlgorithmicNode(node_id='orbit',name='Circular orbital radius and speed',
        description='Infer positive orbital radius from period in the Newtonian circular test-mass model.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PRIMITIVE,inputs=inputs,outputs=outputs)],edges=[],
        metadata={'artifact_source':'physics_reconstructed_numerical_realization','publication_status':'draft',
                  'source_version_id':SOURCE_VERSION,'source_content_hash':SOURCE_HASH,'literal_source_parity':False,
                  'num_nodes':1,'num_edges':0,
                  'physical_regime':'Uniform circular test-mass exterior spherical gravity; geostationary interpretation additionally requires equatorial prograde orbit and sidereal period. Caller establishes clearance.'})
