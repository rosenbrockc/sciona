"""Executable periodic-wave parameter relations."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.physics_ingest.wave_relations_proof import SOURCE_VERSION, SOURCE_HASH

PRIMITIVE = 'sciona.atoms.physics.wave_relations.wave_relations'


def build_wave_relations_execution():
    inputs=[IOSpec(name=n,type_desc='numpy.ndarray',dim_signature=d,constraints=c) for n,d,c in [
        ('frequency','T-1','Positive finite ordinary frequency in cycles/s; identical nonempty input shapes, no broadcasting.'),
        ('phase_speed','L1 T-1','Positive finite phase-speed magnitude in m/s for the same single periodic mode.')]]
    outputs=[IOSpec(name=n,type_desc='numpy.ndarray',dim_signature=d,constraints=c) for n,d,c in [
        ('period','T1','Positive finite period in seconds; input shape.'),('wavelength','L1','Positive finite wavelength in meters; input shape.'),
        ('angular_frequency','T-1','Positive finite angular frequency in rad/s; input shape.'),
        ('angular_wavenumber','L-1','Positive finite angular wavenumber magnitude in rad/m; not cycles/m or signed wavevector.')]]
    return CDGExport(nodes=[AlgorithmicNode(node_id='parameters',name='Periodic wave parameters',
        description='Full source frequency, period, wavelength and angular relations under reviewed positive-domain semantics.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PRIMITIVE,inputs=inputs,outputs=outputs)],edges=[],
        metadata={'artifact_source':'physics_reconstructed_numerical_realization','publication_status':'draft',
                  'source_version_id':SOURCE_VERSION,'source_content_hash':SOURCE_HASH,'literal_source_parity':False,
                  'num_nodes':1,'num_edges':0,'physical_regime':'Single periodic mode, positive ordinary frequency and phase-speed magnitude; no group-velocity or dispersion inference.'})
