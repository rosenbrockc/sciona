"""Executable periodic-wave parameter relations."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.physics_ingest.brewster_angle_proof import SOURCE_VERSION, SOURCE_HASH

PRIMITIVE = 'sciona.atoms.physics.brewster_angle.brewster_angle'


def build_brewster_angle_execution():
    inputs=[IOSpec(name=n,type_desc='numpy.ndarray',dim_signature='1',constraints=c) for n,c in [
        ('incident_index','Positive finite incident-medium index; identical nonempty input shapes.'),
        ('transmitted_index','Positive finite transmitted-medium index; lossless isotropic nonmagnetic interface.')]]
    outputs=[IOSpec(name=n,type_desc='numpy.ndarray',dim_signature='1',constraints=c) for n,c in [
        ('incidence_angle','Radians from normal; atan2(n2,n1), independently rounded. Equal indices: pi/4 convention, not unique Brewster angle.'),
        ('refraction_angle','Radians from normal; atan2(n1,n2), independently rounded; near-grazing may round to float64 pi/2.')]]
    return CDGExport(nodes=[AlgorithmicNode(node_id='interface',name='Brewster incidence and refraction',
        description='Correct source medium ratio with independent complementary-angle computation.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PRIMITIVE,inputs=inputs,outputs=outputs)],edges=[],
        metadata={'artifact_source':'physics_reconstructed_numerical_realization','publication_status':'draft',
                  'source_version_id':SOURCE_VERSION,'source_content_hash':SOURCE_HASH,'literal_source_parity':False,
                  'num_nodes':1,'num_edges':0,'physical_regime':'Planar lossless isotropic nonmagnetic positive-index interface, p-polarization. Rounded angles do not certify exact Fresnel equality.'})
