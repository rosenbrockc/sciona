"""Single-species Langmuir equilibrium from kinetic coefficients."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,IOSpec,NodeStatus
from sciona.physics_ingest.langmuir_proof import SOURCE_VERSION,SOURCE_HASH
PRIMITIVE='sciona.atoms.physics.langmuir.langmuir'


def build_langmuir_execution():
    inputs=[IOSpec(name=n,type_desc='numpy.ndarray',dim_signature=d,constraints=c) for n,d,c in [
        ('adsorption_coefficient','M^-1 L T','Positive finite ka in Pa^-1 s^-1; identical nonempty shapes.'),
        ('desorption_coefficient','T^-1','Positive finite kd at fixed temperature.'),
        ('pressure','M L^-1 T^-2','Nonnegative finite partial pressure in Pa; zero boundary independently validated.'),
        ('total_sites','L^-2','Positive finite total surface site density in m^-2.')]]
    outputs=[IOSpec(name=n,type_desc='numpy.ndarray',dim_signature=d,constraints=c) for n,d,c in [
        ('coverage','1','Fraction of occupied sites; may round to1 near saturation.'),
        ('occupied_sites','L^-2','Occupied site density independently rounded.'),
        ('vacant_sites','L^-2','Vacant site density independently evaluated, not N*(1-rounded coverage).'),
        ('equilibrium_rate','L^-2 T^-1','Common adsorption/desorption event rate; not net accumulation.')]]
    return CDGExport(nodes=[AlgorithmicNode(node_id='equilibrium',name='Langmuir adsorption equilibrium',
        description='Solve rate/site balance with independent vacancy and zero-pressure handling.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PRIMITIVE,inputs=inputs,outputs=outputs)],edges=[],
        metadata={'artifact_source':'physics_reconstructed_numerical_realization','publication_status':'draft',
                  'source_version_id':SOURCE_VERSION,'source_content_hash':SOURCE_HASH,'literal_source_parity':False,
                  'num_nodes':1,'num_edges':0,
                  'physical_regime':'Fixed-temperature single non-dissociative species, identical independent monolayer sites, equilibrium. No lateral interactions or competitive adsorption.'})
