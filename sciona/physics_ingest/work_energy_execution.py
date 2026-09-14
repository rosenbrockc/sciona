"""Executable endpoint kinetic energies and net work."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.physics_ingest.work_energy_proof import SOURCE_VERSION, SOURCE_HASH

PRIMITIVE = 'sciona.atoms.physics.work_energy.work_energy'


def build_work_energy_execution():
    inputs=[IOSpec(name=n,type_desc='numpy.ndarray',dim_signature=d,constraints=c) for n,d,c in [
        ('mass','M1','Positive finite constant mass in kg; identical nonempty shapes for all inputs.'),
        ('initial_velocity','L1 T-1','Finite signed initial 1D velocity in m/s; no broadcasting.'),
        ('final_velocity','L1 T-1','Finite signed final 1D velocity in m/s; caller establishes endpoint consistency.')]]
    outputs=[IOSpec(name=n,type_desc='numpy.ndarray',dim_signature='M1 L2 T-2',constraints=c) for n,c in [
        ('initial_kinetic_energy','Initial kinetic energy in J; nonnegative, input shape.'),
        ('final_kinetic_energy','Final kinetic energy in J; nonnegative, input shape.'),
        ('net_work','Signed net translational work in J; independently rounded exact energy difference.')]]
    return CDGExport(nodes=[AlgorithmicNode(node_id='energy',name='Endpoint work and kinetic energies',
        description='Constant-mass 1D work-energy realization with explicit work reference and corrected source endpoint identities.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PRIMITIVE,inputs=inputs,outputs=outputs)],edges=[],
        metadata={'artifact_source':'physics_reconstructed_numerical_realization','publication_status':'draft',
                  'source_version_id':SOURCE_VERSION,'source_content_hash':SOURCE_HASH,'literal_source_parity':False,
                  'num_nodes':1,'num_edges':0,'physical_regime':'Classical constant positive mass, signed 1D endpoints; source constant net force. No trajectory certification or individual-force-only work claim.'})
