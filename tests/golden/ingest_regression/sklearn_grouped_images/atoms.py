def extract_patches_2d(image, patch_size):
    return {"patches": [], "patch_size": patch_size}


def reconstruct_from_patches_2d(patches, image_size):
    return {"image": [], "image_size": image_size}


def img_to_graph(image):
    return {"graph": "image_graph"}


def grid_to_graph(n_x, n_y, n_z=1):
    return {"graph": "grid_graph", "shape": (n_x, n_y, n_z)}
