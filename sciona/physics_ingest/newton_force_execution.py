"""Conditional Newtonian force law; incomplete source derivation not certified."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,IOSpec,NodeStatus
from sciona.physics_ingest.newton_force_proof import SOURCE_VERSION,SOURCE_HASH
PRIMITIVE='sciona.atoms.physics.newton_force.newton_force'


def build_newton_force_execution():
    inputs=[IOSpec(name=n,type_desc='numpy.ndarray',dim_signature=d,constraints=c) for n,d,c in [
        ('gravitational_constant','L^3 M^-1 T^-2','Positive finite G; constitutive parameter, not derived from F=ma.'),
        ('mass_1','M','Positive finite mass1 in kg; identical nonempty input shapes.'),
        ('mass_2','M','Positive finite mass2 in kg.'),
        ('separation','L','Positive finite center separation in m; caller establishes exterior spherical/point-mass applicability.')]]
    outputs=[IOSpec(name=n,type_desc='numpy.ndarray',dim_signature=d,constraints=c) for n,d,c in [
        ('force_magnitude','M L T^-2','Positive attractive force magnitude; no direction vector.'),
        ('acceleration_1','L T^-2','Positive acceleration magnitude of body1 due to body2.'),
        ('acceleration_2','L T^-2','Positive acceleration magnitude of body2 due to body1.'),
        ('potential_energy','M L^2 T^-2','Negative interaction potential energy with zero at infinity.')]]
    return CDGExport(nodes=[AlgorithmicNode(node_id='interaction',name='Conditional Newtonian gravitational interaction',
        description='Evaluate assumed Newtonian force law and related accelerations/energy; retain source inference gaps.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PRIMITIVE,inputs=inputs,outputs=outputs)],edges=[],
        metadata={'artifact_source':'physics_reconstructed_numerical_realization','publication_status':'draft',
                  'source_version_id':SOURCE_VERSION,'source_content_hash':SOURCE_HASH,'literal_source_parity':False,
                  'num_nodes':1,'num_edges':0,
                  'source_derivation_valid_without_added_assumptions':False,
                  'physical_regime':'Assumed Newtonian point-mass or exterior non-overlapping spherical-body law. Source mass-scaling and inverse-square inference require added premises; no universal-gravitation derivation claim.'})
