import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from ufl import grad
from ufl import inner, dot, avg, jump, dx, FacetNormal, Measure
from ufl import TrialFunction, TestFunction
import dolfinx.la as la
from dolfinx.mesh import Mesh
from dolfinx.fem import Function, functionspace, form, assemble_scalar, ElementMetaData
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.mesh import meshtags, locate_entities_boundary

from sim.common.mesh import locate_boundary_cells, circumcenters, is_weakly_acute

"""
Usage example:
bash sim/common/grad_dg0_unit_test.sh
"""


def discrete_gradient_def(grad_d_FS: functionspace, tau_test: TestFunction, d_FS: functionspace, d_trial: TrialFunction, boundary_conditions: list, normal_F: FacetNormal,  ds: Measure, dS: Measure, cells: np.ndarray = None, int_points: np.ndarray = None):
    """
    Returns the Equation defining the discrete lifting or reconstructed gradient for piecewise constant functions.

    Parameters
    ----------
    grad_d_FS: functionspace
        DG0 Function Space for the reconstructed gradient of a DG0 function.
    tau_test: TestFunction
        TestFunction of the above Function Space.
    d_FS: functionspace
        DG0 Function Space.
    d_trial: TrialFunction
        TrialFunction of the above Function Space.
    boundary_conditions: list
        List of boundary conditions given as objects of the classes described in meta_bcs.py
    normal_F: FacetNormal
        FacetNormal of the domain.
    ds, dS: Measure
        Measures for the outside and interior facets evtl. with subdomain data prescribed.
    cells: np.ndarray
        1D array giving the indices of the cells that need interpolation for the boundary conditions
    int_points: np.ndarray
        of the shape (dim, #cells) yielding the coordinates of the interpolation points for the provdided cells for the boundary conditions

    Returns
    ----------
    matrix_form: variational form, see the following Equation
    vector_form: variational form, see the following Equation
    
    ----------
    The abstract Equation:
        inner(grad_d, tau_test)*dx
        =
        d_trial.x *matrix_form * tau_test.x 
        +
        vector_form * tau_test.x
    """
    
    if cells is None or int_points is None:
        """
        Interpolation points for the interpolation of the boundary condition should be supplied, since they must correspond to the mesh.
        If not, set default interpolation points:
        triangles => circumcenter as default interpolation points
        squares, cubes / otherwise => midpoint as default interpolation point, which is what dolfinx assumes by default
        """
        domain = grad_d_FS.mesh
        if domain.ufl_domain().ufl_cell().cellname() == "triangle":
            cells, int_points = circumcenters(domain)
        elif domain.ufl_domain().ufl_cell().cellname() == "prism":            
            raise ValueError("Prisms are not fully supported in dolfinx v0.9.0.")
        else:
            cells      = list(range(domain.topology.index_map(domain.topology.dim).size_local))
            int_points = d_FS.tabulate_dof_coordinates()


    
    

    matrix_form = inner(dot(avg(tau_test),normal_F('-')),jump(d_trial))*dS    # on the interior
    # NOTE - using the normal in direction '-' is consistent with the Definition of the discrete gradient and is necessary for the right results!

    zero_matrix = Function(grad_d_FS)
    vector_form = inner(zero_matrix, tau_test)*dx
    
    for bcwofs in boundary_conditions:
        if bcwofs.quantity == "d":
            """
            The interpolation points for the boundary condition need to be moved. They are chosen as point-symmetric point of the interpolation point with respect to the boundary facet's barycenter.
            """
            cells_for_bc, int_points_for_bc = compute_dg0_int_pts_on_bdry(d_FS.mesh, cells, int_points, marker = bcwofs.marker)
            # TODO - implement option for topological location of boundary dofs

            # NOTE - meshtag = 0 provided the whole domain in the past, but that does not seem to work reliably anymore
            if bcwofs.meshtag != 0:
                dG = ds(bcwofs.meshtag)
                # NOTE - ds needs to be given as argument due to the meshtags being initialized prior
            else:
                dG = ds

            if bcwofs.type == "Dirichlet": 
                d_bc = Function(d_FS) # create Function for boundary condition
                interpolate_dg0_at(d_bc, bcwofs.values, cells_for_bc, int_points_for_bc, marker = bcwofs.marker) 
                d_bc.x.scatter_forward() 
                
                matrix_form -= 0.5*inner(dot(tau_test,normal_F), d_trial )*dG
                vector_form += 0.5*inner(dot(tau_test,normal_F), d_bc)*dG
            elif bcwofs.type == "Neumann": 
                matrix_form -= inner(dot(tau_test,normal_F), d_trial )*dG  

    return matrix_form, vector_form

def d_to_grad_d_mappings(grad_d_FS: functionspace, grad_d_trial: TrialFunction, tau_test: TestFunction, d_FS: functionspace, d_trial: TrialFunction, boundary_conditions: list, normal_F: FacetNormal, ds: Measure ,dS: Measure , cells = None, int_points = None ):
    """
    Returns the assembled Matrix B and Vector res for the stationary Equation that is a result of the definition of the discrete gradient:
    grad_d.x = B d.x + res

    Parameters
    ----------
    grad_d_FS: functionspace
        DG0 Function Space for the reconstructed gradient of a DG0 function.
    tau_test: TestFunction
        TestFunction of the above Function Space.
    d_FS: functionspace
        DG0 Function Space.
    d_trial: TrialFunction
        TrialFunction of the above Function Space.
    boundary_conditions: list
        List of boundary conditions given as objects of the classes described in meta_bcs.py
    normal_F: FacetNormal
        FacetNormal of the domain.
    ds, dS: Measure
        Measures for the outside and interior facets evtl. with subdomain data prescribed.
    cells: np.ndarray
        1D array giving the indices of the cells that need interpolation for the boundary conditions
    int_points: np.ndarray
        of the shape (dim, #cells) yielding the coordinates of the interpolation points for the provdided cells for the boundary conditions

    Returns
    ----------
    B: PETSc matrix
    b_bc: PETSc vector
    
    ----------
    From the definition of the reconstructed gradient, we have:
    M*grad_d.x = B d.x + b_bc
    """
    M = assemble_matrix(form( inner(grad_d_trial , tau_test)*dx )) 
    M.assemble()
    
    B_form, b_bc_form = discrete_gradient_def(grad_d_FS, tau_test, d_FS, d_trial, boundary_conditions, normal_F,  ds ,dS , cells = cells, int_points = int_points )
        
    B = assemble_matrix(form(B_form))
    B.assemble()   

    b_bc = assemble_vector(form(b_bc_form))
    # fem.petsc.apply_lifting(b, [bilinear_form], bcs=[bcs]) # NOTE - LIFTING NOT NECESSARY HERE SINCE NO BOUNDARY CONDITION NEEDS TO BE IMPOSED
    b_bc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE) # REVIEW - reverse ghost update here necessary? I would expect: yes

    """
    Left multiplying with the inverse matrix of M yields:
    grad_d.x = M^-1*(B d.x + d_bc) = D*(B d.x + d_bc)
    """
    D = M.getDiagonal()     # extract diagonal values: D = diag(M)    
    D.reciprocal()          # invert diagonal values   D <- [1/d_i]_i
    """
    Rescale B to map the unknowns of u onto the unknowns of nabla_u
    B <- D*B
    grad_d.x = B d.x + D*d_bc = B d.x + res
    """
    B.diagonalScale(D, None)        # LEFT MULTIPLICATION BY THE INVERTED DIAGONAL MATRIX FROM 
    b_bc.pointwiseMult(D, b_bc)     # b_bc <- D*b_bc
    b_bc.assemble()                 # update on all ranks
    # REVIEW - reverse ghost update here necessary?

    return B, b_bc
    
def reconstruct_grad(B,res, grad_d: Function, d: Function) -> None:
    """
    Computes the reconstructed gradient of function d based on the matrix B and vector res and saves it in the function grad_d:
    grad_d.x = B d.x + res

    Parameters
    ----------
    B: PETSc matrix
        Should be given by the method d_to_grad_d_mappings.
    res: PETSc vector
        Should be given by the method d_to_grad_d_mappings.
    grad_d: Function
        DG0 Function that will be overwritten with the reconstructed gradient.
    d: Function
        DG0 Function on which the reconstructed gradient is based on.    
    """
    
    # create petsc vectors
    d.x.scatter_forward()
    d_petsc       = la.create_petsc_vector_wrap(d.x)
    grad_d_petsc  = la.create_petsc_vector_wrap(grad_d.x)
    
    B.mult(d_petsc,grad_d_petsc)    # grad_d_petsc = B * d_petsc
    # grad_d_petsc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE) # REVIEW - reverse update here necessary?
    grad_d_petsc.axpy(1.0, res)     # grad_d_petsc += res
    grad_d.x.scatter_forward()      # update all ranks

def interpolate_dg0_at(u: Function, f: callable, cells: np.ndarray, int_pts: np.ndarray, marker: callable = None) -> None:
    """
    Interpolation of the Function u with Callable f at the interpolation points int_pts prescribed to the cells cells. If a geometric marker is provided, the interpolation will be restrained to the marked cells.

    Parameters:
    u: Function
        Dolfinx function to be interpolated.
    f: callable
        Expression to be interpolated.
    cells: np.ndarray
        Cell indices corresponding the interpolation points. (1D array)
    int_pts: np.ndarray
        of the shape (#cells, dim) yielding the coordinates of the interpolation points for the provdided cell indices
    marker: callable 
        Geometric marker as in dirichlet_bc in dolfinx.
    """
    FS = u.function_space
    if not( marker is None):
        # reduce to marked cells
        cells = np.array(cells)
        marked_cells = locate_boundary_cells(FS.mesh, marker = marker)  # NOTE - this function does not return ghost cells
        relevant_cells = np.intersect1d(marked_cells, cells)
        index_relevant_cells = np.isin(cells, relevant_cells)           # get index of relevant cells in our given cells
        assert (cells[index_relevant_cells] == relevant_cells).all()    # sanity test
        int_pts = int_pts[index_relevant_cells]                         # now reduce int_pts to relevant set
        cells = relevant_cells


    """
    In DG0 only one dof is given per cell / cell_component.
    We compute the mapping that maps cell -> cell dofs (shape = (cells, dim)).
    In dolfinx the dofs are given from [0, ... , blocksize * #cells].
    """
    blocksize       = FS.dofmap.index_map_bs
    n_cells         = FS.mesh.topology.index_map(FS.mesh.topology.dim).size_local #NOTE - this EXCLUDES ghost cells
    dofs_per_cell   = np.arange(n_cells*blocksize).reshape((-1, blocksize ))
    # NOTE - this could go wrong if the DG0 space is constructed using a MixedElement instead of a BlockedElement...
    # because for a blocked element the space yields the dofs [1, ..., m] but really having the dofs [1, ..., blocksize * m]
    # while for a mixed element space the mixed space would yield the union of the dofs of the subspace
    dofs            = dofs_per_cell[cells, :]   # compute the dofs that are considered during this interpolation, shape before flattening = (cells, dofmap blocksize)
    # NOTE - in the line above it is important that the cells given as input do NOT include any ghost cells otherwise this will throw an index error
    values          = f(int_pts.T)              # evaluate callable expression to obtain values, int_pts.shape = (cells, dim), values.shape = (dim, cells)
    u.x.array[dofs] = values.T                  # ACTUAL interpolation of the function
    u.x.scatter_forward()                       # update all ranks

def compute_dg0_int_pts_on_bdry(domain: Mesh, cells: np.ndarray, int_points: np.ndarray, marker: callable = None):
    """
    Takes the input interpolation points, considers only the one in the area considered by the geometric marker, and then maps them at artificial nodes that are point-symmetric to the current interpolation point with respect to the boundary facet's barycenter.

    Parameters:
    domain: Mesh
        Object from dolfinx.mesh.Mesh class.
    cells: np.ndarray
        Cell indices corresponding the interpolation points. (1D array)
    int_pts: np.ndarray
        of the shape (#cells, dim) yielding the coordinates of the interpolation points for the provdided cell indices
    marker: callable 
        Geometric marker as in dirichlet_bc in dolfinx.
    
    Returns:
    b_cells: 1D np.ndarray
        Cell indices owned by the process for the boundary condition.
    b_int_points: np.ndarray
        of the shape (#b_cells, dim) yielding the coordinates of the interpolation points for the boundary condition for the provided cell indices in b_cells
    """
    tdim = domain.topology.dim # = number of vertices - 1, relevant for connectivity
    fdim = tdim-1 #facet dimension
    if tdim not in (2, 3) or domain.ufl_domain().ufl_cell().cellname() not in ["tetrahedron", "triangle", "quadrilateral", "hexahedron"]:
        raise ValueError("This routine handles triangles (dim=2) and tetrahedra (dim=3) or quadrilaterals (dim = 2) and hexahedra (dim = 3) only.")
        # NOTE - Prisms e.g. have facets with different amount of nodes. This is not reflected in this method.
    
    coords              = domain.geometry.x                             # vertices coordinates
    domain.topology.create_connectivity(tdim,0)
    c2n                 = domain.topology.connectivity(tdim, 0)         # adjacency list cells -> nodes in cell # NOTE this includes ghost cells
    n_nodes_per_cell    = len(c2n.links(0))                             # local number of nodes per cell
    nodes_per_cell      = np.reshape(c2n.array, (-1,n_nodes_per_cell))  # transform adjacency list to 2d array; [ [node1, node2, node3, ..], [node2, node3, node4, ..], ...]
    
    domain.topology.create_connectivity(fdim,0)
    f2n                 = domain.topology.connectivity(fdim, 0) # adjacency list facet -> nodes in facet
    n_nodes_per_facet   = len(f2n.links(0))                     # local number of facets per cell
    
    marked_boundary_cells = locate_boundary_cells(domain, marker = marker)  # list of indices # NOTE this EXCLUDES ghost cells
    marked_boundary_nodes = locate_entities_boundary(domain, 0, marker)     # list of indices # NOTE this EXCLUDES ghost cells, "Compute mesh entities that are connected to an owned boundary facet and satisfy a geometric marking function."
    
    # NOTE - We assume here that every marked boundary cell has exactly 1 (!) marked boundary facet, otherwise we would have an irregular array leading to an error
    nodes_per_marked_boundary_cell                  = nodes_per_cell[marked_boundary_cells]  # reduce to marked boundary cells # NOTE - here the remaining ghost cell indices are kicked out
    marked_boundary_nodes_per_marked_boundary_cell  = np.reshape(nodes_per_marked_boundary_cell[np.isin(nodes_per_marked_boundary_cell, marked_boundary_nodes)], (-1, n_nodes_per_facet)) # reduce to marked boundary nodes
    marked_boundary_coords_per_marked_boundary_cell = coords[ marked_boundary_nodes_per_marked_boundary_cell ] # compute coordinates for these nodes

    # Compute the barycenter
    # NOTE - Again, we assume here that every marked boundary cell has exactly 1 (!) marked boundary facet
    barycenter_per_marked_boundary_cell = np.average(marked_boundary_coords_per_marked_boundary_cell, axis =1)
    # Compute the projected boundary cells
    int_points[marked_boundary_cells] = barycenter_per_marked_boundary_cell + (barycenter_per_marked_boundary_cell - int_points[marked_boundary_cells])
    # only output relevant cells and interpolation points
    b_cells, b_int_points = np.array(cells)[marked_boundary_cells], int_points[marked_boundary_cells]

    return b_cells, b_int_points

def check_dg0_grad_central_flux(comm: MPI.Comm, domain: Mesh, cells: np.ndarray, int_points: np.ndarray) -> None:
    """
    Checks whether the central flux criterion is fulfilled for a mesh, i.e. for
        x_F         the barycenter of a facet
        x_T1, x_T2  the interpolation points of the cells sharing that facet
        it holds
        x_F = 0.5 * (x_T1 + x_T2)
    """
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    f2c      = domain.topology.connectivity(fdim, tdim)   # facets to cell connection
    n_facets = domain.topology.index_map(fdim).size_local # number of facets

    f2v                 = domain.topology.connectivity(fdim, 0)     # facets to vertex connection
    n_nodes_per_facet   = len(f2v.links(0))                             # local number of nodes per cell
    nodes_per_facet     = np.reshape(f2v.array, (-1,n_nodes_per_facet))
    coords              = domain.geometry.x
    coords_per_facet    = coords[nodes_per_facet]

    # RETRIEVE INTERIOR FACETS
    all_facets                      = np.array(range(n_facets))
    nr_nodes_per_facet              = f2c.offsets[1:]-f2c.offsets[:-1]               
    interior_facets                 = all_facets[nr_nodes_per_facet.astype(int) == 2]

    # RETRIEVE INTERPOLATION POINTS OF FACET SHARING CELLS
    cells_per_interior_facet    = f2c.array[np.ravel([f2c.offsets[interior_facets], f2c.offsets[interior_facets]+1], "F")].reshape((-1, 2)) 
    # NOTE - np.ravel creates alternating index here
    relevant_facets             = interior_facets[np.isin(cells_per_interior_facet, cells).all(axis=1)] # these are interior facets with both cells provided be the function argument
    
    if not np.allclose(relevant_facets, interior_facets):
        raise Warning("Less cells (and interpolation points) were provided than the amount that exists in the mesh.")

    # compute average interpolation point
    avg_int_pts   = np.average(coords_per_facet[relevant_facets], axis = 1)

    # compute barycenter
    barycenters   = np.average(coords_per_facet[relevant_facets], axis=1)

    return np.allclose(avg_int_pts, barycenters)

def test_dg0_grad_approx(comm: MPI.Comm, uex: callable, p: int, celltype:str, diagonal:str, n_min: int, n_max: int, use_bcs: bool):
    """
    Tests the approximation of the reconstructed gradient against an interpolation of the polynomial order p on the mesh resolutions 2**n_min, 2**(n_min + 1), ... , 2**(n_max).

    Parameters
    ----------
    comm: MPI.COMM
    uex: callable
        Function expression to evaluate.
    p: int
        polynomial order of continuous FEM space in which the function expression is interpolated in as comparison.
    celltype: str
        CellType of the mesh.
    diagonal:
        Type of triangulation.
    n_min: int
        Coarsest mesh resolution.
    n_max: int
        Finest mesh resolution.
    use_bcs: bool
        Whether boundary conditions should be computed or not.
    """

    # Triangulation type
    diagonal_dict = {"left_right": DiagonalType.left_right,"right_left": DiagonalType.right_left, "crossed": DiagonalType.crossed, "left": DiagonalType.left, "right": DiagonalType.right}
    if comm.rank == 0 and celltype != "triangle": print("DiagonalType will be ignored for other than triangulations.")

    
    errs = []
    hs   = []
    for i in range(n_min,n_max):
        dh = 2**i
        hs.append(1/dh)
        
        # CREATE DOMAIN
        """
        For linear convergence of our reconstructed gradient, we implicitly assume a form of a central flux.
        The average of two neighbouring interpolation points of our DG0 space must be the barycenter of the shared facet.
        For structured meshes, in particular for cartesian grids, we can expect this for triangles, squares, cubes and prism.
        For tetraeder this cannot be expected.
        """
        if celltype == "triangle":
            dim = 2
            domain = create_unit_square(comm, dh, dh, CellType.triangle, diagonal=diagonal_dict[diagonal])

        elif celltype == "square":
            dim = 2
            domain = create_unit_square(comm, dh, dh, CellType.quadrilateral)

        elif celltype == "cube":
            dim = 3
            domain = create_unit_cube(comm, dh, dh, dh, CellType.hexahedron)
        
        elif celltype == "prism":
            dim = 3
            raise ValueError("Prisms are not fully supported in dolfinx v0.9.0.")
            # domain = create_unit_cube(comm, dh, dh, dh, CellType.prism)
        
               
        # DEFINITION OF BOUNDARIES
        def on_left(x):   return np.isclose(x[0], 0.0)
        def on_right(x):  return np.isclose(x[0], 1.0)
        def on_bottom(x): return np.isclose(x[1], 0.0)
        def on_top(x):    return np.isclose(x[1], 1.0)
        
        if dim == 3:            
            def on_front(x):   return np.isclose(x[1], 0.0)
            def on_back(x):  return np.isclose(x[0], 1.0)
            def boundary(x: np.ndarray) -> np.ndarray:
                return np.logical_or.reduce((np.isclose(x[0], 0.0), np.isclose(x[0], 1.0), np.isclose(x[1], 0.0), np.isclose(x[1], 1.0), np.isclose(x[2], 0.0), np.isclose(x[2], 1.0)))
        else:
            def boundary(x: np.ndarray) -> np.ndarray:
                return np.logical_or.reduce((np.isclose(x[0], 0.0), np.isclose(x[0], 1.0), np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)))
        
        """
        with XDMFFile(comm, "output/testmesh.xdmf", "w") as file:
            file.write_mesh(domain)
        """
        
        # FUNCTION SPACES
        normal_F = FacetNormal(domain)
        ds = Measure("ds", domain=domain)
        dS = Measure("dS", domain=domain)
        
        
        CG   = functionspace(domain, ElementMetaData("Lagrange", max(p,2), shape=(dim,)))   # vector function space exactly resolve function
        vDG0 = functionspace(domain, ElementMetaData("DG", 0, shape=(dim,)))                # vector function space dg0
        tDG0 = functionspace(domain, ElementMetaData("DG", 0, shape = (dim,dim)))           # tensor function space dg0
    
        uexh = Function(CG)         # (exact) interpolation of uex              
        uh = Function(vDG0)         # DG0 interpolation of uex
        nabla_uh = Function(tDG0)   # DG0 interpolation of gradient
        
        H10_integral = inner(grad(uexh) - nabla_uh ,grad(uexh) - nabla_uh)*dx # errornorm integral
        
        # INTERPOLATION
        uexh.interpolate(uex)       # interpolation in the continuous FEM space
        uexh.x.scatter_forward() 

        if celltype == "triangle":
            """
            For right-angled isosceles triangles, we can choose the circumcenter as interpolation point.
            For a crossed triangulation the interpolation points will agree with the facet barycenter of the square that contains four triangles.
            For a left_right, right_left, left, right triangulation the interpolation points will agree with the midpoint of the square that contains two triangles.
            """
            cells, int_points = circumcenters(domain)
            interpolate_dg0_at(uh, uex, cells, int_points)
            uh.x.scatter_forward()

            print("Central Flux Criterion fulfilled:", check_dg0_grad_central_flux(comm, domain, cells, int_points))
            print("Mesh is weakly acute:", is_weakly_acute(domain))

        elif celltype == "prism":
            """
            As interpolation point one would need to take the average of the triangles' barycenter.
            """
            raise ValueError("Prisms are not fully supported in dolfinx v0.9.0.")
            
        else:
            """
            For squares and cubes the midpoint is already set as interpolation point by dolfinx.
            """            
            uh.interpolate(uex)
            uh.x.scatter_forward()
            cells, int_points = None, None

            print(f"Central Flux Criterion fulfilled n={i}:", check_dg0_grad_central_flux(comm, domain, list(range(domain.topology.index_map(domain.topology.dim).size_local)), vDG0.tabulate_dof_coordinates() ))
            print("Mesh is weakly acute:", is_weakly_acute(domain))
        
        # BOUNDARY CONDITIONS
        if use_bcs: 
            if celltype != "triangle" or diagonal in ["right_left", "right", "left"] or dim == 3:
                """
                This case: (some) corner cells have more than one boundary facet
                => we have to define a boundary function on each boundary part to circumvent this issue
                """

                # Retrieve facet indices
                facets_left   = locate_entities_boundary(domain, dim-1, on_left)
                facets_right  = locate_entities_boundary(domain, dim-1, on_right)
                facets_bottom = locate_entities_boundary(domain, dim-1, on_bottom)
                facets_top    = locate_entities_boundary(domain, dim-1, on_top)
                all_facets= [facets_left, facets_right, facets_bottom, facets_top]
                all_marker = [
                                np.full(facets_left.size,   1, dtype=np.int32),
                                np.full(facets_right.size,  2, dtype=np.int32),
                                np.full(facets_bottom.size, 3, dtype=np.int32),
                                np.full(facets_top.size,    4, dtype=np.int32),
                            ]

                if dim == 3:
                    facets_front = locate_entities_boundary(domain, dim-1, on_front)
                    facets_back  = locate_entities_boundary(domain, dim-1, on_back)
                    all_facets += [facets_front, facets_back]
                    all_marker += [np.full(facets_front.size,   5, dtype=np.int32), np.full(facets_back.size,   6, dtype=np.int32)]
                
                # Create Mapping:
                facet_indices = np.concatenate(all_facets).astype(np.int32)
                marker_array  = np.concatenate(all_marker)
                # Create Meshtags:
                domain_boundaries = meshtags(domain, dim-1, facet_indices, marker_array)
                # Create Measure with supplied meshtags
                ds = Measure("ds", domain=domain, subdomain_data=domain_boundaries) # overwrite existing Measure
                # Create Dirichlet boundary conditions
                bcs = [
                    meta_dirichletbc( "d", "geometric", uex, marker = on_left, meshtag=1 ),
                    meta_dirichletbc( "d", "geometric", uex, marker = on_right, meshtag=2 ),
                    meta_dirichletbc( "d", "geometric", uex, marker = on_bottom, meshtag=3 ),
                    meta_dirichletbc( "d", "geometric", uex, marker = on_top, meshtag=4 ),
                    ]
                if dim == 3:
                    bcs += [meta_dirichletbc( "d", "geometric", uex, marker = on_front, meshtag=5 ), meta_dirichletbc( "d", "geometric", uex, marker = on_back, meshtag=6 )]

            else:
                """
                Simplest case:
                => Every boundary cell has at most one facet on the boundary
                """
                bcs = [meta_dirichletbc( "d", "geometric", uex, marker = boundary )]

        else: 
            """
            Assembly of boundary conditions is excluded via cli argument.
            """
            bcs = []

        # COMPUTATION OF THE DG0 GRADIENT
        B, res = d_to_grad_d_mappings(tDG0, TrialFunction(tDG0), TestFunction(tDG0), vDG0 , TrialFunction(vDG0),  bcs, normal_F, ds ,dS , cells = cells, int_points = int_points)
        reconstruct_grad(B,res, nabla_uh, uh)

        # ERROR COMPUTATION
        err_local   = assemble_scalar(form(H10_integral))
        # print(f"Rank {comm.rank}: Local H^1_0-integral: {err_local}") # only for debugging
        err_global = comm.allreduce(err_local, op=MPI.SUM)**0.5

        errs.append(err_global)
        # Only print the error on one process
        # if comm.rank == 0:
        #     print(f"Error_H10 (DG0 expl) for n = {dh} : {err_global:.2e}")

    rates = np.log(np.array(errs[1:]) / np.array(errs[:-1])) / np.log(np.array(hs[1:]) / np.array(hs[:-1]))
    print("Errors:", errs, "\nfor mesh resolutions: ", hs)
    print("\n\n ==> Convergence Rates: ", rates, "\n\n")

if __name__ == "__main__":
    import argparse
    import numpy as np
    from ufl import grad, Mesh, Measure
    from dolfinx.io import XDMFFile
    from dolfinx.mesh import create_unit_square, create_unit_cube, CellType
    from dolfinx.cpp.mesh import DiagonalType
    from dolfinx.fem import functionspace
    from basix.ufl import element
    from sim.common.norms import L2_norm
    # from sim.common.error_computation import errornorm
    from sim.common.meta_bcs import meta_dirichletbc
    """
    Test case:
    """
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    if rank == 0: print(f"{comm.size} MPI ranks activated")

    parser = argparse.ArgumentParser(description="Test case for approximation with reconstructed gradient in DG0.")
    
    parser.add_argument('-ct','--celltype', type=str, choices=["triangle", "square", "cube", "prism"], default="simplical", help = "For the domain decomposition choose between triangles (dim = 2), squares (dim = 2), cubes (dim = 3) and prisms (dim = 3). NOTE: Prisms are currently not fully supported in dolfinx 0.9.0.")
    parser.add_argument('-diag','--diagonal', type=str, choices=["crossed", "left_right", "right_left", "left", "right"], default="crossed", help='Type of triangulation. (only for 2D)')
    parser.add_argument('-n_min', type=int, default = 1, help='2**n is the coarsest mesh resolution considered.')
    parser.add_argument('-n_max', type=int, default = 5, help='2**n is the finest mesh resolution considered.')
    parser.add_argument('-nobcs', action='store_const', default=False, const=True, help = "Leave out assembly of boundary conditions.")


    args = parser.parse_args()

    if args.celltype in ["triangle", "square"]: 
        dim = 2
    else:
        dim = 3

    if rank == 0: print(f"Input: {args.__dict__}")


    #SECTION - TEST CASES
    # POLYNOMIAL ORDER
    for p in [2,3,4,5]:
        if rank == 0: print(f"--- Approximation of the function: f(x) = (x_1^{p}, -x_2^{p}, 0) ---")
        def exmpl(x):
            values = np.zeros((dim, x.shape[1])) 

            values[0] = x[0]**p
            values[0] = (-1)*x[1]**p

            return values
        
        test_dg0_grad_approx(comm, exmpl, p, args.celltype, args.diagonal, args.n_min, args.n_max, not (args.nobcs))
    # OTHER EXAMPLES
    for i in range(1,4,1):
        if rank == 0: print(f"--- Approximation of the function: f(x) = (cos(A * x_1), sin(A * x_2), 0) with frequency A = {2*i} * pi ---")
        def exmpl(x):
            values = np.zeros((dim, x.shape[1])) 

            values[0] = np.cos(i*np.pi*x[0])
            values[1] = np.sin(i*np.pi*x[1])

            return values
        
        test_dg0_grad_approx(comm, exmpl, 4, args.celltype, args.diagonal, args.n_min, args.n_max, not (args.nobcs))
    # """
    #!SECTION

    """
    12/08/2025: unit-test shows linear order convergence as expected in serial
    """