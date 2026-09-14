import ast
import builtins
import numpy as np
import pytest
import sympy as sp
from sciona.physics_ingest.equation_runtime import compile_equation_runtime


def runtime():
    a,b,current,total=sp.symbols('a b current total')
    return compile_equation_runtime(sp.Eq(total,a+b),[sp.Ne(current,0),sp.Ge(a,0),sp.Ge(b,0)])


def load(compiled):
    def guarded_import(name,*args,**kwargs):
        if name!='numpy':raise AssertionError('unexpected runtime import')
        return builtins.__import__(name,*args,**kwargs)
    namespace={'__builtins__':{**vars(builtins),'__import__':guarded_import}}
    exec(compiled.source,namespace)
    return namespace['evaluate']


def test_numpy_only_runtime_and_broadcasting():
    compiled=runtime();fn=load(compiled)
    assert compiled.argument_symbols==('a','b','current')
    np.testing.assert_allclose(fn(np.array([[2.],[5.]]),np.array([3.,7.]),1.),[[5.,9.],[8.,12.]])
    assert all(n.module!='sympy' for n in ast.walk(ast.parse(compiled.source)) if isinstance(n,ast.ImportFrom))


@pytest.mark.parametrize('args',[(2.,3.,0.),(-2.,3.,1.),(2.,np.nan,1.),(True,3.,1.),('2',3.,1.),(2+1j,3.,1.)])
def test_invalid_inputs_and_conditions_are_rejected(args):
    with pytest.raises(ValueError):load(runtime())(*args)


def test_overflow_is_not_returned_as_a_valid_prediction():
    with pytest.raises(FloatingPointError):load(runtime())(1.7e308,1.7e308,1.)


def test_symbol_names_do_not_become_python_code():
    symbol=sp.Symbol("unsafe name'); import os")
    compiled=compile_equation_runtime(sp.Eq(sp.Symbol('out'),symbol+1))
    assert 'unsafe' not in compiled.source
    assert load(compiled)(2)==3


def test_implicit_equations_and_functions_are_rejected():
    x,y=sp.symbols('x y')
    with pytest.raises(ValueError):compile_equation_runtime(sp.Eq(x+y,1))
    with pytest.raises(ValueError):compile_equation_runtime(sp.Eq(y,sp.sin(x)))
