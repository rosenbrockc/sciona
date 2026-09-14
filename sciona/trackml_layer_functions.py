"""TrackML layer-function interpolation; BSD-2-Clause source adaptation.

Derived from edwinst/trackml_solution; see docs/licenses/TrackML-BSD-2-Clause.txt.
Grid values are supplied at runtime; no detector maps are bundled.
Source linear extrapolation and leading singleton-dimension removal are retained.
"""
from types import SimpleNamespace
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def createLayerFunctionInterpolators(is_cylinder, layer_id, layer_functions=None):
    if layer_functions is not None:
        interpolators = {}
        fns = layer_functions.functions
        fdata = layer_functions.getGridValues(is_cylinder=is_cylinder, layer_id=layer_id, functions=fns)
        for name, (points, values) in zip(fns, fdata):
            if points is None:
                interpolators[name] = (0, 0, None)
            else:
                ndropped_dims = 0
                while len(points) > 1 and len(points[0]) == 1:
                    ndropped_dims += 1
                    points = points[1:]
                    values = values.reshape(values.shape[1:])
                for coord in points:
                    assert len(coord) > 1
                interpolators[name] = (len(points), ndropped_dims, RegularGridInterpolator(points=points, values=values, method='linear', bounds_error=False, fill_value=None))
    else:
        interpolators = None
    return interpolators

class CylinderLayer:

    def evaluateLayerFunctions(self, x, y, z, functions, points=None):
        """Evaluate per-layer interpolated functions at the given coordinates.
            Args:
                x, y, z (float array(N,)): coordinates at which to evaluate the functions.
                    Note: Ignored if `points` is given.
                functions (list of str): names of the functions to evaluate.
                points (None or array(N,M)): If given, evaluate layer functions at the
                    given point coordinates and ignore (x, y, z) arguments.
            Returns:
                vals (list of float array(N,)): for each name passed in `functions`,
                    this list containes the interpolated values of the corresponding
                    layer function for the given coordinates.
            """
        vals = []
        phi_z = None
        if self.layer_functions is not None:
            for fn in functions:
                val = None
                nargs, ndrop, interpolator = self.layer_functions[fn]
                if interpolator is not None:
                    if points is not None:
                        arg = points[:, ndrop:]
                    elif nargs == 1:
                        assert ndrop == 0
                        arg = z
                    else:
                        assert ndrop == 0
                        if phi_z is None:
                            phi_z = np.stack([z, np.arctan2(y, x)], axis=1)
                        arg = phi_z
                    val = interpolator(arg)
                vals.append(val)
        return vals

class CapLayer:

    def evaluateLayerFunctions(self, x, y, z, functions, points=None):
        """Evaluate per-layer interpolated functions at the given coordinates.
            Args:
                x, y, z (float array(N,)): coordinates at which to evaluate the functions.
                    Note: Ignored if `points` is given.
                functions (list of str): names of the functions to evaluate.
                points (None or array(N,M)): If given, evaluate layer functions at the
                    given point coordinates and ignore (x, y, z) arguments.
            Returns:
                vals (list of float array(N,)): for each name passed in `functions`,
                    this list containes the interpolated values of the corresponding
                    layer function for the given coordinates.
            """
        vals = []
        r2 = None
        phi_r2 = None
        if self.layer_functions is not None:
            for fn in functions:
                val = None
                nargs, ndrop, interpolator = self.layer_functions[fn]
                if interpolator is not None:
                    if points is not None:
                        arg = points[:, ndrop:]
                    elif nargs == 1:
                        assert ndrop == 0
                        if r2 is None:
                            r2 = np.sqrt(np.square(x) + np.square(y))
                        arg = r2
                    else:
                        assert ndrop == 0
                        if phi_r2 is None:
                            r2 = np.sqrt(np.square(x) + np.square(y))
                            phi_r2 = np.stack([np.arctan2(y, x), r2], axis=1)
                        arg = phi_r2
                    val = interpolator(arg)
                vals.append(val)
        return vals

class LayerFunctionEvaluator:

    def evaluateLayerFunctions(self, xi, yi, zi, cyl_closer, next_id, functions=None, points=None):
        """Evaluate per-layer interpolated functions at the given coordinates.
        Args:
            xi, yi, zi (float array(N,)): coordinates at which to evaluate the functions.
                Note: Ignored if `points` is given.
            cyl_closer (bool or bool array(N,)): True if the respective point is on a
                cylinder layer, False if it is on a cap.
            next_id (int array(N,)): id of the layer the respective point is on
                (cyl_id for cyl_closer==True, cap_id for cyl_closer==False).
            functions (list of str): names of the functions to evaluate.
                If `None`, evaluate all available layer functions.
            points (None or array(N,M)): If given, evaluate layer functions at the
                given point coordinates and ignore (xi, yi, zi) arguments.
        Returns:
            vals (list of float array(N,)): for each name passed in `functions`,
                this list containes the interpolated values of the corresponding
                layer function for the given coordinates.
        """
        assert self.layer_functions is not None
        if functions is None:
            functions = self.layer_functions.functions
        if points is not None:
            n = points.shape[0]
            int_finite = True
        else:
            n = len(xi)
            int_finite = np.isfinite(xi) & np.isfinite(yi) & np.isfinite(zi)
        values = [np.full(n, np.nan, dtype=np.float64) for fn in functions]
        cyl_finite = int_finite & cyl_closer
        selection = cyl_finite
        ids = np.unique(next_id[selection])
        for id in ids:
            selection_id = selection & (next_id == id)
            lay = self.cyln.cylinders[id]
            if points is not None:
                val = lay.evaluateLayerFunctions(None, None, None, functions, points=points[selection_id, :])
            else:
                val = lay.evaluateLayerFunctions(xi[selection_id], yi[selection_id], zi[selection_id], functions)
            for dst, src in zip(values, val):
                if src is not None:
                    dst[selection_id] = src
        cap_finite = int_finite & np.logical_not(cyl_closer)
        selection = cap_finite
        ids = np.unique(next_id[selection])
        for id in ids:
            selection_id = selection & (next_id == id)
            lay = self.capn.caps[id]
            if points is not None:
                val = lay.evaluateLayerFunctions(None, None, None, functions, points=points[selection_id, :])
            else:
                val = lay.evaluateLayerFunctions(xi[selection_id], yi[selection_id], zi[selection_id], functions)
            for dst, src in zip(values, val):
                if src is not None:
                    dst[selection_id] = src
        return values

def build_layer_evaluator(layer_functions, *, cylinder_count, cap_count):
    """Build source interpolation dispatch without requiring fitted hit neighbors."""
    evaluator = LayerFunctionEvaluator()
    evaluator.layer_functions = layer_functions
    groups = []
    for cls, count, is_cylinder in [(CylinderLayer, cylinder_count, True), (CapLayer, cap_count, False)]:
        layers = []
        for layer_id in range(count):
            layer = cls()
            layer.layer_functions = createLayerFunctionInterpolators(is_cylinder, layer_id, layer_functions)
            layers.append(layer)
        groups.append(layers)
    evaluator.cyln = SimpleNamespace(cylinders=groups[0])
    evaluator.capn = SimpleNamespace(caps=groups[1])
    return evaluator
