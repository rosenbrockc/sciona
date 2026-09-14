"""Independent central-difference oracle on synthetic neural inputs."""
import numpy as np
from sciona.flavours_neural import loss_gradients


def test_all_parameter_gradients_match_central_difference():
    rng = np.random.default_rng(21)
    x = rng.normal(size=(7, 3))
    y = np.arange(7) % 2
    params = [rng.normal(size=(3, 8)), rng.normal(size=8),
              rng.normal(size=(8, 2)), rng.normal(size=2)]
    _, gradients = loss_gradients(x, y, params)
    for p, grad in zip(params, gradients):
        for index in np.ndindex(p.shape):
            original = p[index]
            p[index] = original + 1e-6
            plus = loss_gradients(x, y, params)[0]
            p[index] = original - 1e-6
            minus = loss_gradients(x, y, params)[0]
            p[index] = original
            np.testing.assert_allclose(grad[index], (plus-minus)/2e-6, atol=1e-8, rtol=1e-6)
