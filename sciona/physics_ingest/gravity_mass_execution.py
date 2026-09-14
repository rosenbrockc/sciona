"""Executable periodic-wave parameter relations."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.physics_ingest.gravity_mass_proof import SOURCE_VERSION, SOURCE_HASH

PRIMITIVE = 'sciona.atoms.physics.gravity_mass.gravity_mass'


def build_gravity_mass_execution():
    inputs=[IOSpec(name=n,type_desc='numpy.ndarray',dim_signature=d,constraints=c) for n,d,c in [
        ('acceleration','L1 T-2','Positive finite pure gravitational acceleration in m/s².'),
        ('radius','L1','Positive finite center distance in meters at surface/exterior of spherical source.'),
        ('gravitational_constant','M-1 L3 T-2','Positive finite G in m³/kg/s²; identical nonempty shapes, no broadcasting.')]]
    outputs=[IOSpec(name='mass',type_desc='numpy.ndarray',dim_signature='M1',constraints='Positive finite inferred mass in kg; exact arithmetic before rounding.')]
    return CDGExport(nodes=[AlgorithmicNode(node_id='mass_inference',name='Infer spherical-source gravitational mass',
        description='Infer M=g*r²/G for pure gravitational acceleration, retaining the reviewed source-example discrepancy.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PRIMITIVE,inputs=inputs,outputs=outputs)],edges=[],
        metadata={'artifact_source':'physics_reconstructed_numerical_realization','publication_status':'draft',
                  'source_version_id':SOURCE_VERSION,'source_content_hash':SOURCE_HASH,'literal_source_parity':False,
                  'num_nodes':1,'num_edges':0,'physical_regime':'Newtonian spherical-source exterior/surface field. Pure gravitational acceleration, not rotation-adjusted effective gravity. No precision Earth ephemeris claim.'})
