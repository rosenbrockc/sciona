"""Conditional sound-speed scaling with retained approximation diagnostics."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.physics_ingest.sound_scale_proof import SOURCE_VERSION,SOURCE_HASH

PRIMITIVE='sciona.atoms.physics.sound_scale.sound_scale'


def build_sound_scale_execution():
    inputs=[IOSpec(name=n,type_desc='numpy.ndarray',dim_signature=d,constraints=c)
            for n,d,c in [
                ('atomic_mass','1','Dimensionless m/mp >=1; identical nonempty input shapes.'),
                ('fine_structure','1','Positive finite fine-structure constant.'),
                ('light_speed','L T^-1','Positive finite light speed in m/s.'),
                ('electron_mass','M','Positive finite electron mass in kg.'),
                ('proton_mass','M','Positive finite proton mass in kg.'),
                ('bulk_prefactor','1','Positive finite f in K=f*E/a^3; exposes dropped prefactor.'),
                ('shear_to_bulk','1','Nonnegative finite G/K; exposes bulk-only approximation.')]]
    outputs=[IOSpec(name=n,type_desc='numpy.ndarray',dim_signature=d,constraints=c)
             for n,d,c in [
                 ('speed_estimate','L T^-1','Unit-prefactor bulk-only Rydberg scaling estimate; not material certification.'),
                 ('upper_scale','L T^-1','A=1 maximum of unit-prefactor scaling curve only; not universal physical bound.'),
                 ('bulk_model_speed','L T^-1','Retains sqrt(f) under Rydberg bonding-energy model.'),
                 ('longitudinal_model_speed','L T^-1','Retains prefactor and shear; may exceed upper_scale.'),
                 ('shear_factor','1','sqrt(1+4*(G/K)/3), diagnostic of omitted shear contribution.')]]
    return CDGExport(nodes=[AlgorithmicNode(node_id='scaling',name='Conditional sound-speed scaling',
        description='Correct denominator2 and retain prefactor/shear diagnostics under explicit bonding-energy model.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PRIMITIVE,
        inputs=inputs,outputs=outputs)],edges=[],metadata={
            'artifact_source':'physics_reconstructed_numerical_realization','publication_status':'draft',
            'source_version_id':SOURCE_VERSION,'source_content_hash':SOURCE_HASH,'literal_source_parity':False,
            'num_nodes':1,'num_edges':0,
            'physical_regime':'Conditional Rydberg bonding-energy scale; isotropic longitudinal elasticity; A>=1. No universal material-speed bound proven.'})
