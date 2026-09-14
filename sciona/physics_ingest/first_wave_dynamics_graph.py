"""Corrected differential scope with explicit premises; original graph untouched."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus


def build_first_wave_dynamics_graph():
    node=AlgorithmicNode(node_id='dynamics',name='Constant-mass differential dynamics',
        description='Apply explicit acceleration/position kinematics to Newton second law, solve acceleration and optionally sample.',
        concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.physics.first_wave_dynamics.first_wave_dynamics',
        inputs=[IOSpec(name='payload',type_desc='dict')],outputs=[IOSpec(name='result',type_desc='dict')])
    return CDGExport(nodes=[node],edges=[],metadata=dict(
        artifact_source='physics_corrected_execution',publication_status='draft',
        source_version_id='693697cf-a97c-53bd-8d6b-19dc3c7697c1',
        source_content_hash='97dbc4000139b6514d678829e16c39e045f02f47ecbe91356863028decb932b3',
        scope='One inertial Cartesian component; symbolic acceleration=x second derivative, force=mass*acceleration and acceleration=force/mass; optional point evaluation.',
        corrections=['Replace invalid ordinary-symbol derivative parse with actual symbolic differentiation',
            'Add missing acceleration/position kinematic premise','Require positive constant mass'],
        runtime_contract='Version1 dict: finite positive mass, explicit arithmetic/function position tree or formal x(t), finite sample times.',
        assumptions=['Twice differentiable position on evaluation domain','Consistent SI coefficients','Inertial Cartesian coordinate'],
        exclusions=['Time integration','Global smoothness certification','Original malformed proof certification','Tier1 human certification'],
        num_nodes=1,num_edges=0))
