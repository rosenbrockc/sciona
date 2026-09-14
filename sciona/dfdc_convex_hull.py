"""DFDC outline-polygon occlusion with explicit detector/predictor dependencies.

MIT 2020 Selim Seferbekov, 89c6290490bac96b29193a4061b3db9dd3933e36;
see docs/licenses/DFDC-MIT.txt. Source function body is unchanged. The original
name refers to an ordered landmark polygon, not a computed mathematical hull.
The supplied predictor must expose the source 68-point landmark convention.
This module does not load detector models or predictor weights.
"""

import random
import numpy as np
import skimage.draw
from skimage import measure

def blackout_convex_hull(img, detector, predictor):
    try:
        rect = detector(img)[0]
        sp = predictor(img, rect)
        landmarks = np.array([[p.x, p.y] for p in sp.parts()])
        outline = landmarks[[*range(17), *range(26, 16, -1)]]
        Y, X = skimage.draw.polygon(outline[:, 1], outline[:, 0])
        cropped_img = np.zeros(img.shape[:2], dtype=np.uint8)
        cropped_img[Y, X] = 1
        y, x = measure.centroid(cropped_img)
        y = int(y)
        x = int(x)
        first = random.random() > 0.5
        if random.random() > 0.5:
            if first:
                cropped_img[:y, :] = 0
            else:
                cropped_img[y:, :] = 0
        elif first:
            cropped_img[:, :x] = 0
        else:
            cropped_img[:, x:] = 0
        img[cropped_img > 0] = 0
    except Exception as e:
        pass
