import numpy as np
from dolfinx.mesh import exterior_facet_indices, locate_entities_boundary, Mesh

"""
Helper Functions to work with mesh geometry and topology.
"""

def locate_boundary_cells(domain: Mesh, marker: callable = None):
    """
    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
    marker: callable
        geometric marker to identify a subdomain 

    Returns
    -------
    marked_boundary_cells: np.ndarray
        Indices of marked cells that are owned by the process with boundary facets. (1d array)
    """
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    f2c     = domain.topology.connectivity(fdim, tdim)  # facets to cell connection  # NOTE - this includes ghosts
    bfacets = exterior_facet_indices(domain.topology)   # indices of boundary facets # NOTE - this EXCLUDES ghosts "Compute the indices of all exterior facets that are owned by the caller."
    if not (marker is None):
        bfacets = locate_entities_boundary(domain, fdim, marker)    # marked boundary facets
        # bfacets = np.intersect1d(bfacets, mfacets)                # not necessary  

    # For exterior facets, links has length 1, since only one cell can own an exterior facet
    marked_boundary_cells = np.unique(f2c.array[f2c.offsets[bfacets]])

    return marked_boundary_cells

def circumcenters(domain: Mesh):
    """
    Compute circumcenters for all cells of a simplicial mesh.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        Mesh with triangles (2D) or tetrahedra (3D).

    Returns
    -------
    cells: np.ndarray
        Indices of cells. (1d array with length n_cells)
    centers : (n, 3) ndarray
        Circumcenter coordinates for each cell.
    """
    tdim = domain.topology.dim
    gdim = domain.geometry.dim

    if tdim not in (2, 3) or domain.ufl_domain().ufl_cell().cellname() not in ["triangle", "tetrahedron"]:
        raise ValueError("This routine handles triangles (tdim=2) and tetrahedra (tdim=3) only.")

    # COMPUTE VERTICES AND CELL-VERTEX CONNECTION
    domain.topology.create_connectivity(tdim, 0)                # Need cell->vertex connectivity #NOTE - this includes the ghost cells!
    c2v                 = domain.topology.connectivity(tdim, 0)         # create adjacency list mapping cells -> list of cell vertices
    n_cells             = domain.topology.index_map(tdim).size_local    # number of all locally owned cells #NOTE - this EXCLUDES the ghost cells
    cells               = list(range(n_cells))
    vertices            = domain.geometry.x                     # coordinates array (num_vertices, gdim)
    nodes_per_cell      = np.reshape(c2v.array, (-1,gdim+1))    # NOTE - we here assumed a simplical element
    nodes_per_cell      = nodes_per_cell[:n_cells]              # Removal of Ghost Cells 
    # NOTE - the local ordering is always: cells owned by the local process 0: index_map.size_local, ghost cells index_map.size_local +1 : ....
    vertices_per_cell   = vertices[nodes_per_cell]
    
    """
    The calculation works as followed:
    For each cell with vertices [v0, v1, ...]:
        # compute edge vectors
        A_j         := v_j - v0 , j = 1, ...
        # compute barycenters
        bc_j        := (v_j + v_0 ) / 2 , j = 1, ...
        # the center c must fulfill:
        A_j . (c - bc_j) = 0
    <=> A_j . c = A_j . bc_j := b_j
        # This must be fulfilled since the difference between center and the facet barycenter is a scaled facet normal.
        # The facet normal however must be normal to the facet edge.

        # the right hand side can be computed explicitly
    =>  b_j         =  (|v_j|^2 + |v_0|^2 ) / 2 , j = 1, ...
        # we need to solve the linear equation system given by A, b
    """
    
    A_per_cell = vertices_per_cell[:, 1:gdim+1, 0:gdim] - vertices_per_cell[:, 0:1, 0:gdim] 
    # NOTE - Using 0:1 in the second array is necessary to preserve the middle dimension
    # NOTE - The slicing 0:gdim in the last dimension is necessary to get rid of the third coordinate in the 2D case

    b_per_cell = 0.5 * ( np.linalg.norm(vertices_per_cell[:,1:gdim+1,0:gdim], axis = 2)**2 - np.linalg.norm(vertices_per_cell[:, 0:1, 0:gdim], axis = 2)**2 )

    centers = np.empty((n_cells, 3)) # always 3 dimensional points with zero for third dimension
    
    # NOTE - converting everything into a block diagonal works and takes less lines, but is much more memory and CPU intensiv than looping through the cells. 
    # This method takes about a minute for 10^6 nodes, which is sufficient for my case
    # NOTE - this could be improved by parallelizing the loop
    for i in range(n_cells):
        A_i      = A_per_cell[i]
        b_i      = b_per_cell[i]
        center_i = np.linalg.solve(A_i, b_i)       
              
        if gdim == 2:
            centers[i,:] = np.hstack([center_i,0]) 
        elif gdim == 3:
            centers[i,:] = center_i 
        else:
            raise ValueError("This routine handles dim = 2,3 only.")

    return cells, centers

def is_weakly_acute(domain: Mesh) -> bool:
    """
    Computes whether a domain is weakly acute, i.e. all angles between two facets are less than or equal to 90 degrees.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        Mesh with triangles (2D) or tetrahedra (3D).

    Returns
    -------
    bool: whether it is true.
    """
    gdim = domain.geometry.dim
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, 0)
    f2v                     = domain.topology.connectivity(fdim, 0)     # facets to vertex connection
    n_nodes_per_facet       = len(f2v.links(0))                         # local number of nodes per cell
    nodes_per_facet         = np.reshape(f2v.array, (-1,n_nodes_per_facet))
    coords                  = domain.geometry.x
    coords_per_facet        = coords[nodes_per_facet]
    barycenter_per_facet    = np.average(coords_per_facet, axis = 1)

    # Compute facet normal
    if gdim == 2:
        edges_per_facet     = coords_per_facet[:, 1, 0:gdim] - coords_per_facet[:, 0, 0:gdim] 
        R                   = np.array([ [0, 1], [-1, 0] ])    # Rotation matrix
        normal_per_facet    =  np.matvec(R, edges_per_facet)   # Applicaiton of rotation matrix
    elif gdim == 3:
        edges_per_facet     = coords_per_facet[:, 1:, 0:gdim] - coords_per_facet[:, 0:1, 0:gdim] 
        normal_per_facet    = np.cross(edges_per_facet[:, 0], edges_per_facet[:, 1])
    # NOTE - in both dimension we only know that it is a scaled version of the normal vector. We do not have any knowledge whether it is the inner or outer normal

    else:
        raise ValueError("Only dimension 2 and 3 supported.")    
    
    # Connect Cells to facets
    domain.topology.create_connectivity(tdim, fdim)
    c2f                             = domain.topology.connectivity(tdim, fdim)      # facets to vertex connection
    n_facets_per_cell               = len(c2f.links(0))                             # local number of nodes per cell
    facets_per_cell                 = np.reshape(c2f.array, (-1,n_facets_per_cell)) # shape = (#cells, #facets per cell)
    barycenter_per_facet_per_cell   = barycenter_per_facet[facets_per_cell] # shape = (#cells, #facets per cell, 3)

    # Connect Cells to vertices
    domain.topology.create_connectivity(tdim, 0)
    c2v                 = domain.topology.connectivity(tdim, 0)     # facets to vertex connection
    n_nodes_per_cell    = len(c2v.links(0))                             # local number of nodes per cell
    nodes_per_cell      = np.reshape(c2v.array, (-1,n_nodes_per_cell))
    coords_per_cell     = coords[nodes_per_cell]
    midpoint_per_cell   = np.average(coords_per_cell, axis = 1) # shape = (#cells, 3)

    # COMPUTE OUTER NORMAL
    # In order to do so we take the midpoint of the cell and substract the barycenter.
    # We obtain the midpoint relative to the facet. We can now use the outer normal as a seperating hyperplane.
    # If the midpoint lies "inside" with respect to the normal, then its inner product is negative. Otherwise its positive.
    # Accordingly we switch the sign of the normal, to obtain the outer normal.
    diff_midpoint_barycenter_per_facet_per_cell = midpoint_per_cell[:, np.newaxis, 0:gdim] - barycenter_per_facet_per_cell[:, :, 0:gdim]    # shape = (#cells, #facets per cell, gdim)
    normal_per_facet_per_cell                   = normal_per_facet[facets_per_cell]                                                         # shape = (#cells, #facets per cell, gdim)
    outer_normal_per_facet_per_cell             = normal_per_facet_per_cell * np.einsum("ijk,ijk->ij", normal_per_facet_per_cell, diff_midpoint_barycenter_per_facet_per_cell)[:, :, np.newaxis] * (-1) # shape = (#cells, #facets per cell, gdim)

    # COMPUTE WEAKLY ACUTE CRITERION
    # If a convex cell is weakly acute, all its facet angles are 90 degrees or less.
    # Equivalently, the inner product of the outer normals is non-positive.
    # We compute the inner product of the outer normals
    facet_normal_product_per_cell = np.einsum("ijk,ilk->ijl",outer_normal_per_facet_per_cell, outer_normal_per_facet_per_cell)
    # we delete the diagonal where the values are of course always positive.
    i, j = np.indices((n_facets_per_cell, n_facets_per_cell))
    facet_normal_product_per_cell_no_diag = facet_normal_product_per_cell[:, i != j].reshape((-1,n_facets_per_cell, n_facets_per_cell-1 ))
    # now we can evaluate the whole mesh
    bool_weakly_acute = (facet_normal_product_per_cell_no_diag <= 0).all()

    return bool_weakly_acute

def midpoints(domain: Mesh):
    """
    Compute midpoints for all cells of a mesh.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        Mesh

    Returns
    -------
    cells: np.ndarray
        Indices of cells. (1d array with length n_cells)
    centers : (n, 3) ndarray
        Midpoint coordinates for each cell.
    """
    tdim = domain.topology.dim
    gdim = domain.geometry.dim


    # COMPUTE VERTICES AND CELL-VERTEX CONNECTION
    domain.topology.create_connectivity(tdim, 0)                # Need cell->vertex connectivity and local cell count
    c2v         = domain.topology.connectivity(tdim, 0)         # create adjacency list mapping cells -> list of cell vertices
    n_vertices  = domain.topology.index_map(tdim).size_local    # number of all locally owned vertices

    vertices            = domain.geometry.x                     # coordinates array (num_vertices, gdim)
    n_nodes_per_cell    = len(c2v.links(0))   
    nodes_per_cell      = np.reshape(c2v.array, (-1,n_nodes_per_cell))    
    vertices_per_cell   = vertices[nodes_per_cell]
    centers             = np.average(vertices_per_cell, axis=1)        
    n_cells             = len(vertices_per_cell)
    cells               = list(range(n_cells))
    
    return cells, centers

    

if __name__ == "__main__":
    from mpi4py import MPI
    from dolfinx.mesh import create_unit_square, create_unit_cube, CellType
    from dolfinx.cpp.mesh import DiagonalType
    import time
    """
    Test case for circumcenters

    NOTE: The test cases are not MPI safe.
    """

    # A square consisting of two triangles with the diagonal from the top left to the bottom right
    # +-------+
    # |\      |
    # | \     |
    # |  \    |
    # |   \   |
    # |    \  |
    # |     \ |
    # |      \|
    # +-------+
    domain = create_unit_square(MPI.COMM_WORLD, 1, 1, CellType.triangle, diagonal=DiagonalType.left)
    cells, centers = circumcenters(domain)
    expected_result = [[0.5, 0.5, 0], [0.5, 0.5, 0]]
    assert np.allclose(centers, expected_result), "Failure for triangle"
    print("Domain 1 fulfills central flux property for circumcenters.")
    print("Domain 1 is weakly acute:", is_weakly_acute(domain))

    # A square consisting of two triangles with the diagonal from the bottom left to the top right
    # +-------+
    # |      /|
    # |     / |
    # |    /  |
    # |   /   |
    # |  /    |
    # | /     |
    # |/      |
    # +-------+
    domain = create_unit_square(MPI.COMM_WORLD, 1, 1, CellType.triangle, diagonal=DiagonalType.right)
    cells, centers = circumcenters(domain)
    expected_result = [[0.5, 0.5, 0], [0.5, 0.5, 0]]
    assert np.allclose(centers, expected_result), "Failure for triangle"
    print("Domain 2 fulfills central flux property for circumcenters.")
    print("Domain 2 is weakly acute:", is_weakly_acute(domain))

    # A square consisting of four right-angled, isosceles triangles
    # +-----+
    # |\   /|
    # | \ / |
    # |  X  |
    # | / \ |
    # |/   \|
    # +-----+
    domain = create_unit_square(MPI.COMM_WORLD, 1, 1, CellType.triangle, diagonal=DiagonalType.crossed)
    cells, centers = circumcenters(domain)
    expected_result = np.array([[0, 0.5, 0], [0.5, 0, 0], [1, 0.5, 0], [0.5, 1, 0]])
    assert np.allclose(np.sort(centers, axis = 0), np.sort(expected_result, axis = 0))
    print("Domain 3 fulfills central flux property for circumcenters.")
    print("Domain 3 is weakly acute:", is_weakly_acute(domain))

    # 3D test case for tetrahedra: A cube consisting of 6(!) tetrahedra
    domain = create_unit_cube(MPI.COMM_WORLD, 1, 1, 1, CellType.tetrahedron)
    cells, centers = circumcenters(domain)
    expected_result = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    assert np.allclose(centers, expected_result), "Failure for tetrahedron"
    print("Domain 4 (3D) fulfills central flux property for circumcenters.")
    print("Domain 4 is weakly acute:", is_weakly_acute(domain))

    print("All sanity tests succesful. Starting time measurement.")

    # Time Measurement for big case
    n = 100
    domain = create_unit_square(MPI.COMM_WORLD, n,n, CellType.triangle, diagonal=DiagonalType.crossed)

    start = time.time()
    cells, centers = circumcenters(domain)
    end = time.time()
    print("elapsed time:", end - start, "; for #nodes =", n**2)



    