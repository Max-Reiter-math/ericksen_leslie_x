import numpy as np
from dolfinx.mesh import locate_entities, meshtags

def set_attributes(object, default_attributes, args):
    """
    Assigns all attributes in dict default_attributes to the instance object. These will by overwriten by dict args for the intersection of their keys
    """
    args_dict = vars(args)
    for key in default_attributes:
        if key in args_dict and args_dict[key]!= None: setattr(object, key, args_dict[key])
        else: setattr(object, key, default_attributes[key])

def retrieve_meshtags(mesh, boundaries):
    """
    boundaries takes list of tuples (marker, locator)
    """
    facet_indices, facet_markers = [], []
    fdim = mesh.topology.dim - 1
    for (marker, locator) in boundaries:
        facets = locate_entities(mesh, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = meshtags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
    return facet_tag

def get_global_dofs(FS):
    num_dofs_global = FS.dofmap.index_map.size_global * FS.dofmap.index_map_bs
    return  num_dofs_global

def get_local_dofs(FS):
    num_dofs_local = FS.dofmap.index_map.size_local * FS.dofmap.index_map_bs
    return num_dofs_local

