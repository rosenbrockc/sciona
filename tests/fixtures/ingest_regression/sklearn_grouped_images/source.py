import numpy as np


def extract_patches_2d(image, patch_size):
    patch_h, patch_w = patch_size
    patches = []
    for row in range(image.shape[0] - patch_h + 1):
        for col in range(image.shape[1] - patch_w + 1):
            patches.append(image[row : row + patch_h, col : col + patch_w])
    return np.asarray(patches)


def reconstruct_from_patches_2d(patches, image_size):
    image_h, image_w = image_size
    patch_h, patch_w = patches.shape[1:]
    canvas = np.zeros((image_h, image_w), dtype=patches.dtype)
    counts = np.zeros((image_h, image_w), dtype=float)
    idx = 0
    for row in range(image_h - patch_h + 1):
        for col in range(image_w - patch_w + 1):
            canvas[row : row + patch_h, col : col + patch_w] += patches[idx]
            counts[row : row + patch_h, col : col + patch_w] += 1.0
            idx += 1
    counts[counts == 0.0] = 1.0
    return canvas / counts


def img_to_graph(image):
    flattened = np.asarray(image).reshape(-1)
    size = int(flattened.shape[0])
    return np.eye(size)


def grid_to_graph(n_x, n_y, n_z=1):
    node_count = int(n_x) * int(n_y) * int(n_z)
    return np.zeros((node_count, node_count))
