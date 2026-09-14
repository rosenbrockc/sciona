"""Numerical realization of the source Schwarzschild length-scale conclusion."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,IOSpec,NodeStatus

PRIMITIVE='sciona.atoms.physical_quantities.schwarzschild_radius.schwarzschild_radius'
SOURCE_VERSION='f05dd820-e273-506d-87a3-625d4cd08e76'
SOURCE_HASH='1281198730365e9b8a3c1fc0f0ac941cd9e82bc4d9ac63964ae30d9a8a0aba01'


def build_schwarzschild_execution():
    return CDGExport(nodes=[AlgorithmicNode(node_id='radius',name='Compute Schwarzschild length scale',
        description='Evaluate the source conclusion 2GM/c^2 for positive SI inputs; no black-hole classification.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PRIMITIVE,
        inputs=[IOSpec(name='mass_kg',type_desc='numpy.ndarray',constraints='nonempty finite positive real mass in kilograms',dim_signature='M1'),
            IOSpec(name='gravitational_constant',type_desc='float',constraints='finite positive scalar G in SI',dim_signature='L3 M-1 T-2'),
            IOSpec(name='speed_of_light',type_desc='float',constraints='finite positive scalar vacuum light speed in SI',dim_signature='L1 T-1')],
        outputs=[IOSpec(name='radius_metres',type_desc='numpy.ndarray',constraints='positive finite float64 metres; mass shape preserved',dim_signature='L1')])],edges=[],
        metadata={'artifact_source':'physics_numerical_realization','publication_status':'draft',
            'source_version_id':SOURCE_VERSION,'source_content_hash':SOURCE_HASH,
            'physical_regime':'Schwarzschild length scale; horizon interpretation limited to spherical nonrotating uncharged asymptotically flat vacuum exterior',
            'num_nodes':1,'num_edges':0})
