"""Execution realization of the corrected circular two-body period proof."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus

PRIMITIVE = 'sciona.atoms.physical_quantities.circular_orbital_period.circular_orbital_period'
SOURCE_VERSION = 'ffb043f3-05a3-5ca6-9d94-87a17ff512d3'
SOURCE_HASH = '4d2810f058da2f5d3ec97b8de34b7219dee8b7c1715d99d37f3544fe3e9df9c4'


def build_two_body_execution():
    return CDGExport(nodes=[AlgorithmicNode(node_id='period', name='Compute circular two-body period',
        description='Compute positive orbital period from separation, both masses and explicit SI G under the isolated Newtonian circular-orbit model.',
        concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=PRIMITIVE,
        inputs=[IOSpec(name='separation_metres', type_desc='numpy.ndarray', constraints='nonempty finite positive real separation in metres', dim_signature='L1'),
                IOSpec(name='mass1_kg', type_desc='numpy.ndarray', constraints='finite positive first mass in kilograms; same shape as separation', dim_signature='M1'),
                IOSpec(name='mass2_kg', type_desc='numpy.ndarray', constraints='finite positive second mass in kilograms; same shape as separation', dim_signature='M1'),
                IOSpec(name='gravitational_constant', type_desc='float', constraints='finite positive scalar G in SI', dim_signature='L3 M-1 T-2')],
        outputs=[IOSpec(name='period_seconds', type_desc='numpy.ndarray', constraints='positive finite float64 seconds; input shape preserved', dim_signature='T1')])], edges=[],
        metadata={'artifact_source': 'physics_corrected_numerical_realization', 'publication_status': 'draft',
                  'source_version_id': SOURCE_VERSION, 'source_content_hash': SOURCE_HASH, 'literal_source_parity': False,
                  'physical_regime': 'Isolated Newtonian point masses in a circular orbit; separation is not a barycentric radius.',
                  'num_nodes': 1, 'num_edges': 0})
