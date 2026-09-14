"""Executable periodic-wave parameter relations."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.physics_ingest.parallel_resistance_proof import SOURCE_VERSION, SOURCE_HASH

PRIMITIVE = 'sciona.atoms.physics.parallel_resistance.parallel_resistance'


def build_parallel_resistance_execution():
    inputs=[IOSpec(name=n,type_desc='numpy.ndarray',dim_signature=d,constraints=c) for n,d,c in [
        ('resistance_1','M1 L2 T-3 I-2','Positive finite first resistance in ohms; same two network nodes.'),
        ('resistance_2','M1 L2 T-3 I-2','Positive finite second resistance in ohms; identical nonempty shapes.'),
        ('voltage','M1 L2 T-3 I-1','Finite signed common voltage in volts; zero allowed, no broadcasting.')]]
    outputs=[IOSpec(name=n,type_desc='numpy.ndarray',dim_signature=d,constraints=c) for n,d,c in [
        ('equivalent_resistance','M1 L2 T-3 I-2','Positive finite equivalent resistance in ohms.'),
        ('current_1','I1','Signed current in first branch, amperes.'),
        ('current_2','I1','Signed current in second branch, amperes.'),
        ('total_current','I1','Signed total current, independently rounded from exact sum.')]]
    return CDGExport(nodes=[AlgorithmicNode(node_id='network',name='Two-resistor parallel network',
        description='Compute equivalent resistance and all branch/total currents without premature rounding.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PRIMITIVE,inputs=inputs,outputs=outputs)],edges=[],
        metadata={'artifact_source':'physics_reconstructed_numerical_realization','publication_status':'draft',
                  'source_version_id':SOURCE_VERSION,'source_content_hash':SOURCE_HASH,'literal_source_parity':False,
                  'num_nodes':1,'num_edges':0,'physical_regime':'Two ideal positive finite linear resistors across common nodes; signed common voltage including zero. No reactive/negative/short/open scope.'})
