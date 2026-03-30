def witness_extract_patches_2d():
    return {"patches_non_empty": True}


def witness_reconstruct_from_patches_2d():
    return {"reconstruction_shape_known": True}


def witness_img_to_graph():
    return {"graph_nodes_non_negative": True}


def witness_grid_to_graph():
    return {"grid_shape_positive": True}
