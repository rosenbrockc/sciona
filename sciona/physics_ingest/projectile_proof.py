"""Explicit correction of projectile source identities and time elimination."""
from dataclasses import dataclass
import sympy as sp
from sciona.ghost.symbolic import serialize_expr

SOURCE_VERSION = '229845ef-5dbc-5617-8485-718c0874d427'
SOURCE_HASH = '128603cb8a840c5c965ab9fc96dc9c0035c30380aaeca76d7b33812d2c2f1ecf'


def symbols():
    return dict(zip(['x', 'x0', 'y', 'y0', 'vx0', 'vy0', 'g', 't'],
                    sp.symbols('x x0 y y0 vx0 vy0 g t', real=True)))


@dataclass(frozen=True)
class ProjectileProof:
    premises: tuple
    steps: tuple


def build_proof():
    s = symbols(); x, x0, y, y0, vx, vy, g, t = s.values()
    time = (x-x0)/vx
    return ProjectileProof((sp.Eq(vx*t, x-x0, evaluate=False), sp.Eq(y, y0+vy*t-g*t*t/2, evaluate=False)),
                           (sp.Eq(t, time, evaluate=False), sp.Eq(y, y0+vy*time-g*time*time/2, evaluate=False)))


def verify_proof(proof):
    expected = build_proof()
    if proof.premises != expected.premises or len(proof.steps) != 2:
        raise ValueError('Corrected initial-velocity kinematic premises and two steps required')
    s = symbols()
    first = sp.Eq(proof.premises[0].lhs/s['vx0'], proof.premises[0].rhs/s['vx0'], evaluate=False)
    if proof.steps[0] != first:
        raise ValueError('Time elimination must divide by nonzero horizontal velocity')
    second = sp.Eq(proof.premises[1].lhs, proof.premises[1].rhs.subs(s['t'], first.rhs), evaluate=False)
    if proof.steps[1] != second or second != expected.steps[1]:
        raise ValueError('Substitution must retain linear gravity and initial vertical velocity')
    return dict(source_version_id=SOURCE_VERSION, source_content_hash=SOURCE_HASH, source_ast_parity=False,
                premises=[serialize_expr(e) for e in proof.premises], steps=[serialize_expr(e) for e in proof.steps],
                assumptions=['vx0 != 0', 'g >= 0', '(x-x0)/vx0 >= 0', 'Constant downward acceleration, no drag, common Cartesian SI frame.'],
                corrections=['Initial vertical velocity uses the source symbol v0y, not instantaneous vy.',
                             'Gravity is linear, not squared, in the eliminated trajectory.'],
                checks=dict(division_under_nonzero_premise=True, time_substitution=True))
