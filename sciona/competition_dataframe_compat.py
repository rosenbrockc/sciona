"""Explicit, bounded compatibility for reviewed legacy DataFrame.as_matrix calls.

Callers must verify source content hashes before compiling. Original source
files and pandas classes are never modified. Supported calls use no arguments
or an explicit columns selection, all of whose labels must exist.
"""
import ast
import pandas as pd

HELPER = '_sciona_reviewed_dataframe_array'


def dataframe_array(frame, columns=None):
    if not isinstance(frame, pd.DataFrame):
        raise ValueError('Reviewed conversion requires a pandas DataFrame')
    if columns is None:
        return frame.to_numpy()
    if isinstance(columns, str):
        raise ValueError('Explicit existing column labels required')
    columns = list(columns)
    if any(c not in frame.columns for c in columns):
        raise ValueError('Explicit existing column labels required')
    return frame.loc[:, columns].to_numpy()


def compile_as_matrix_compat(content, *, expected_calls, filename='<reviewed-source>'):
    tree = ast.parse(content)
    if any(isinstance(n, ast.Name) and n.id == HELPER for n in ast.walk(tree)):
        raise ValueError('Compatibility helper identity collides with source')

    class Rewrite(ast.NodeTransformer):
        count = 0

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if not isinstance(node.func, ast.Attribute) or node.func.attr != 'as_matrix':
                return node
            if len(node.args) > 1 or len(node.keywords) > 1 or any(k.arg != 'columns' for k in node.keywords) or (node.args and node.keywords):
                raise ValueError('Unreviewed as_matrix signature')
            self.count += 1
            return ast.copy_location(ast.Call(func=ast.Name(id=HELPER, ctx=ast.Load()),
                args=[node.func.value, *node.args], keywords=node.keywords), node)

    rewrite = Rewrite()
    tree = rewrite.visit(tree)
    if rewrite.count != expected_calls:
        raise ValueError('Legacy conversion inventory differs')
    ast.fix_missing_locations(tree)
    return compile(tree, filename, 'exec'), {HELPER: dataframe_array}
