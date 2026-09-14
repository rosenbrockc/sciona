"""Build explicit execution realizations using shared registered residual atoms."""
import sympy as sp
from sciona.atoms.electrical import residuals
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,IOSpec,NodeStatus
from sciona.ghost.registry import REGISTRY
from sciona.physics_ingest.residual_equivalence import equivalent_residual_mapping


def residual_templates():
    v,i,r,total,a,b=sp.symbols('voltage current resistance total_voltage voltage_a voltage_b')
    rt,ra,rb=sp.symbols('total_resistance resistance_a resistance_b')
    return [(residuals.ohmic_voltage_residual,sp.Eq(v,i*r,evaluate=False)),
            (residuals.series_voltage_residual,sp.Eq(total,a+b,evaluate=False)),
            (residuals.series_resistance_residual,sp.Eq(rt,ra+rb,evaluate=False)),
            (residuals.series_ohmic_balance_residual,sp.Eq(i*rt,i*ra+i*rb,evaluate=False))]


def build_residual_execution(equation,dimensions,*,source_version_id,source_content_hash):
    """Return a draft CDG and callable-argument to source-symbol mapping."""
    for implementation,template in residual_templates():
        fqdn=implementation.__module__+'.'+implementation.__name__
        dim_map=REGISTRY[fqdn]['dim_signature']
        target_dims={name:value.to_compact() for name,value in dim_map.items() if name!='return'}
        mapping=equivalent_residual_mapping(equation,template,dimensions,target_dims)
        if mapping is None:continue
        arguments={target:source for source,target in mapping.items()}
        graph=CDGExport(nodes=[AlgorithmicNode(
            node_id='residual',name='Evaluate equation residual',concept_type='custom',status=NodeStatus.ATOMIC,
            matched_primitive=fqdn,description='Evaluate LHS minus RHS; does not solve an unknown or establish physical validity.',
            inputs=[IOSpec(name=name,type_desc='numpy.ndarray',constraints='finite real; broadcastable',dim_signature=dim) for name,dim in target_dims.items()],
            outputs=[IOSpec(name='residual',type_desc='numpy.ndarray',dim_signature=dim_map['return'].to_compact())],
        )],edges=[],metadata={'artifact_source':'physics_numerical_realization','publication_status':'draft',
            'source_version_id':str(source_version_id),'source_content_hash':source_content_hash,
            'source_symbol_bindings':arguments,'runtime_kind':'equation_residual_validator','num_nodes':1,'num_edges':0})
        return graph,arguments
    raise ValueError('no dimension-preserving shared residual implementation')
