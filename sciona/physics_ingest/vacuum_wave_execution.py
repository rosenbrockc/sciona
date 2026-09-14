"""Synthesis-time vacuum vector wave-equation realization."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,IOSpec,NodeStatus
from sciona.physics_ingest.vacuum_wave_proof import SOURCE_VERSION,SOURCE_HASH

PRIMITIVE='sciona.atoms.physics.vacuum_wave.vacuum_wave'


def build_vacuum_wave_execution():
    inputs=[IOSpec(name=n,type_desc='str',constraints=c) for n,c in [
        ('field_srepr','Tuple(Ex,Ey,Ez) scalar commutative ASTs, C2 on a common open Cartesian space-time region; SI V/m.'),
        ('coordinates_srepr','Tuple(x,y,z,t) of distinct real Symbols; spatial coordinates in m, time in s.'),
        ('permeability_srepr','Provably positive real scalar AST independent of coordinates; vacuum mu in H/m.'),
        ('permittivity_srepr','Provably positive real scalar AST independent of coordinates; vacuum epsilon in F/m.')]]
    outputs=[IOSpec(name=n,type_desc='str',constraints=c) for n,c in [
        ('laplacian_srepr','Tuple of formal componentwise spatial Laplacians; SI V/m^3.'),
        ('scaled_acceleration_srepr','Tuple of mu*epsilon*second time derivatives; SI V/m^3.'),
        ('divergence_srepr','Formal scalar divergence; SI V/m^2. Must vanish for charge-free Gauss; not inferred from the wave equation.')]]
    return CDGExport(nodes=[AlgorithmicNode(node_id='wave',name='Construct vacuum electric-field wave equation',
        description='Formal vector wave-equation sides and separate Gauss constraint, conditional on vacuum Maxwell premises.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PRIMITIVE,inputs=inputs,outputs=outputs)],edges=[],
        metadata={'artifact_source':'physics_reconstructed_symbolic_realization','publication_status':'draft',
                  'source_version_id':SOURCE_VERSION,'source_content_hash':SOURCE_HASH,'literal_source_parity':False,
                  'num_nodes':1,'num_edges':0,'execution_phase':'symbolic synthesis',
                  'physical_regime':'C2 Cartesian fields, constant positive vacuum mu/epsilon and zero charge/current. Necessary Maxwell consequence, not full compliance certification or PDE solver.'})
