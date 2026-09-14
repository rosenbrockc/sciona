"""Executable local temperature derivative of internal energy at fixed pressure."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,IOSpec,NodeStatus
from sciona.physics_ingest.isobaric_energy_proof import SOURCE_VERSION,SOURCE_HASH

PRIMITIVE='sciona.atoms.physics.isobaric_internal_energy.isobaric_internal_energy'


def build_isobaric_energy_execution():
    ports=[IOSpec(name=name,type_desc='numpy.ndarray',dim_signature=dim,constraints=constraint) for name,dim,constraint in [
        ('cv','M1 L2 T-2 Th-1','Finite real Cv=dU/dT at fixed V in J/K; common state/composition.'),
        ('internal_pressure','M1 L-1 T-2','Finite real pi_T=dU/dV at fixed T in Pa; distinct from mechanical pressure.'),
        ('volume','L3','Finite strictly positive volume in m^3.'),
        ('expansion','Th-1','Finite real alpha=(1/V)dV/dT at fixed p in K^-1; may be signed.')]]
    for port in ports:port.constraints+=' Identical nonempty shapes for all ports; no broadcasting.'
    return CDGExport(nodes=[AlgorithmicNode(node_id='energy_derivative',name='Evaluate isobaric internal-energy derivative',
        description='Cv + pi_T V alpha from the local differential chain rule, with corrected internal-pressure dimensions.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PRIMITIVE,inputs=ports,
        outputs=[IOSpec(name='internal_energy_temperature_derivative',type_desc='numpy.ndarray',dim_signature='M1 L2 T-2 Th-1',
                        constraints='Finite float64 J/K, input shape preserved; dU/dT at fixed pressure, not Cp.')])],edges=[],
        metadata={'artifact_source':'physics_reconstructed_numerical_realization','publication_status':'draft',
                  'source_version_id':SOURCE_VERSION,'source_content_hash':SOURCE_HASH,'literal_source_parity':False,
                  'num_nodes':1,'num_edges':0,'physical_regime':'Consistent local derivatives of differentiable U(T,V) and V(T,p), positive volume, fixed composition/amount.'})
