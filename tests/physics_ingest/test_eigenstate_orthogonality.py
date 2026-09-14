"""Exact synthetic runtime coverage for two eigenstates."""
import pytest
import sympy as sp
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.eigenstate_orthogonality import eigenstate_orthogonality, _checked_parse


def invoke(A,u,v,a,b):
    source=[sp.Tuple(*(sp.Tuple(*row) for row in A)),sp.Tuple(*u),sp.Tuple(*v),sp.sympify(a),sp.sympify(b)]
    return tuple(sp.simplify(_checked_parse(e)) for e in eigenstate_orthogonality(*map(sp.srepr,source)))


def test_complex_distinct():
    assert invoke([[2,sp.I],[-sp.I,2]],[sp.I,1],[-sp.I,1],3,1)==(0,0,0)


def test_degenerate_complex_unnormalized():
    assert invoke([[3,0],[0,3]],[sp.I,2],[2,sp.I],3,3)==(0,0,0)
    assert invoke([[3,0],[0,3]],[sp.I,2],[1,sp.I],3,3)==(sp.I,3*sp.I,0)


def test_scalar_negative_eigenvalue():
    assert invoke([[-2]],[2+sp.I],[3-sp.I],-2,-2)==(5-5*sp.I,-10+10*sp.I,0)


def test_exact_algebraic():
    r=sp.sqrt(2)
    assert invoke([[0,r],[r,0]],[1,1],[1,-1],r,-r)==(0,0,0)


def test_symbolic_common_eigenvalue():
    a=sp.Symbol('a',real=True)
    assert invoke([[a,0],[0,a]],[1,2],[3,4],a,a)==(11,11*a,0)


@pytest.mark.parametrize('n',[2,3,7])
def test_exact_dense_basis(n):
    # Rational orthogonal Householder transform; independent known eigenvectors.
    z=sp.Matrix(range(1,n+1));Q=sp.eye(n)-2*z*z.T/(z.T*z)[0]
    A=Q*sp.diag(*range(1,n+1))*Q.T
    assert invoke(A.tolist(),list(Q[:,0]),list(Q[:,n-1]),1,n)==(0,0,0)


@pytest.mark.parametrize('A,u,v,a,b',[
    ([[1,1],[0,2]],[1,0],[1,1],1,2),
    ([[1,0],[0,2]],[1,0],[1,1],1,2),
    ([[1]],[0],[1],1,1),
    ([[1]],[1],[0],1,1),
    ([[1]],[1],[1],1,2),
    ([[1,0]],[1],[1],1,1),
    ([[1]],[1,2],[1],1,1),
    ([[sp.oo]],[1],[1],1,1),
    ([[sp.nan]],[1],[1],1,1),
    ([[sp.I]],[1],[1],sp.I,sp.I),
    ([[1]],[sp.Symbol('q')],[1],1,1),
    ([[1]],[sp.Symbol('q',real=True)],[1],1,1),
    ([],[],[],1,1),
])
def test_reject_invalid_premises(A,u,v,a,b):
    with pytest.raises(ValueError):invoke(A,u,v,a,b)


@pytest.mark.parametrize('payload',[
    "Integral('__import__(\"os\").getcwd()')", "Symbol('__import__(x)')",
    "Add('1+1',Integer(1))", "__import__('os')", "Float('nan')",
])
def test_guard_rejects_constructor_strings(payload):
    with pytest.raises(ValueError):_checked_parse(payload)
