import warnings
import numpy as np
from mpi4py import MPI
from ufl import TrialFunction, TestFunction, inner, dx, Measure
from ufl.core.expr import Expr
from dolfinx.fem import Function, functionspace, form, Expression, assemble_scalar, ElementMetaData
from dolfinx.fem.petsc  import LinearProblem
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx import geometry
from dolfinx.io import XDMFFile

def mpi_time(comm, start = None):
    comm.Barrier()
    if start == None:
        return MPI.Wtime()
    else:
        elapsed_local = MPI.Wtime() - start
        elapsed_global = comm.allreduce(elapsed_local, op=MPI.MAX)  
        return elapsed_global

def update_and_scatter(list_to_update, list_to_update_with):
    for (u, u_update) in list(zip(list_to_update,list_to_update_with)):
        u_update.x.scatter_forward()
        u.x.array[:] = u_update.x.array[:]
        u.x.scatter_forward()

def scatter_all(list):
    for u in list:
        u.x.scatter_forward()

def nodal_normalization(d, dim):
    d.x.scatter_forward()

    coeffs = np.reshape( d.x.array[:] , (-1, dim)) # has shape (#nodes, dim)
    norms = np.linalg.norm(coeffs, axis=1, keepdims=True).flatten() 
    zero_norm_dofs = np.isclose(norms, 0.0) # do not correct zero values
    coeffs[~zero_norm_dofs, :] = coeffs[~zero_norm_dofs, :] / norms[~zero_norm_dofs, np.newaxis]
    # Overwrite coefficients
    d.x.array[:] = np.reshape(coeffs, (-1,))

    d.x.scatter_forward()

#SECTION - Projections

def project(f: Function, V: functionspace, bcs: list = [])-> Function:
    """
    Standard L^2 projection of f onto the space V.
    """
    Pf = TrialFunction(V)
    v = TestFunction(V)
    U = Function(V)
    L = form(inner(Pf,v)*dx)
    R = form(inner(f,v)*dx)
    problem = LinearProblem(L, R,  bcs=bcs, u=U) 
    problem.solve()
    return U



def project_lumped(f: Function, V: functionspace, bcs: list = [])-> Function:
    """
    Mass-Lumping adjusted L^2 projection of f onto the space V, i.e. this method returns Pf as solution of the equation system:
    (Pf,phi)_h = (f,phi)_2 for all phi \in V.
    """
    u = TrialFunction(V)
    v = TestFunction(V)
    U = Function(V)
    # Mass Lumping
    dxL = Measure("dx", metadata = {"quadrature_rule": "vertex"})
    L = form(inner(u,v)*dxL)
    R = form(inner(f,v)*dx)
    problem = LinearProblem(L, R,  bcs=bcs, u=U) 
    problem.solve()
    return U

def nodal_projection_unit(f: Function)-> Function:
    """
    Normalization of coefficients / nodal values of a function. Only well defined for globally continuous Lagrange functions.
    """
    #TODO - test what happens when a tensor shape is given as input
    fs = f.function_space
    element = fs.ufl_element()
    dim = element.num_sub_elements()
    # coefficients of function in vector form
    coeffs = np.reshape( f.x.array[:] , (-1, dim)) # has shape (#nodes, dim)
    norms = np.linalg.norm(coeffs, axis=1, keepdims=True) 
    normed_coeffs = coeffs / norms
    # Overwrite coefficients
    f.x.array[:] = np.reshape(normed_coeffs, (-1,))
    return f

def Pi_h(f1: Function, f0: Function):
    """
    Manipulation of coefficients / nodal values of a function. 
    Equivalent of the mathematical expression
    f1_ := (f0 + (I-f0\otimes f0) (f1-f0)) , 
    return:
        \mathcal{I}_h (f1_/\abs{f1_})

    Only well defined for globally continuous Lagrange functions.
    """
    #TODO - test what happens when a tensor shape is given as input
    fs = f0.function_space
    element = fs.ufl_element()
    dim = element.num_sub_elements()
    # coefficients of function in vector form
    coeffs_f0 = np.reshape( f0.x.array[:] , (-1, dim)) # has shape (#nodes, dim)
    coeffs_diff = np.reshape( f1.x.array[:] - f0.x.array[:] , (-1, dim)) # has shape (#nodes, dim)
    # Compute (I-f0\otimes f0) (f1-f0))
    coeffs_diff_orth = coeffs_diff - coeffs_f0*(np.einsum('ij, ij->i', coeffs_f0 ,coeffs_diff))[:, None]
    # Compute f1_ = (f0 + (I-f0\otimes f0) (f1-f0))
    coeffs_f1_ = coeffs_f0 + coeffs_diff_orth
    # Normalization
    norms = np.linalg.norm(coeffs_f1_, axis=1, keepdims=True) 
    normed_coeffs = coeffs_f1_ / norms
    # Overwrite coefficients
    f1.x.array[:] = np.reshape(normed_coeffs, (-1,))
    return f1
    

#!SECTION

#SECTION - Local Evaluations
def eval_continuous_function(f: Function, points: np.ndarray)-> np.ndarray:
    """
    Evaluates the function f at the given points.
    The function must be locally continuous at the given points otherwise the function cannot be evaluated. 
    This excludes the evaluation of DGp functions at the nodes.
    """
    #TODO - check shape of points array
    domain = f.function_space.mesh
    # Create a bounding box tree for use in collision detection. Padding seems to be some tolerance parameter.
    tree = bb_tree(domain, domain.topology.dim)
    # Compute collisions between points and leaf bounding boxes. Bounding boxes can overlap, therefore points can collide with more than one box.
    cell_candidates = compute_collisions_points(tree, points.T)
    # From a mesh, find which cells collide with a set of points. 
    colliding_cells = compute_colliding_cells(domain, cell_candidates, points.T)
    # Now select the cells that will be used for the evaluation
    cells = []
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            cells.append(colliding_cells.links(i)[0])
        else:
            warnings.warn("No cell found that contains the point " + str(point))
    # Now we can apply the eval function
    return f.eval(points.T,cells).T

def add_Functions(a: Function, b: Function)-> Function:
    """
    For Functions a,b in the same function space this method returns a+b as Function of that function space.
    """
    fs = a.function_space
    res = Function(fs)
    res.x.array[:] = a.x.array[:] + b.x.array[:]
    return res

def substract_Functions(a: Function, b: Function)-> Function:
    """
    For Functions a,b in the same function space this method returns a-b as Function of that function space.
    """
    fs = a.function_space
    res = Function(fs)
    res.x.array[:] = a.x.array[:] - b.x.array[:]
    return res

def test_unit_norm(f: Function)-> dict:
    """
    return the max, min and average of the nodal norms of function f.
    """
    domain = f.function_space.mesh
    points = domain.geometry.x
    y = eval_continuous_function(f, points.T)
    norms = np.linalg.norm(y, axis=0)
    return {"max": np.max(norms), "min": np.min(norms), "average": np.average(norms)}

def test_ptw_orthogonality(f: Function, g: Function, exclude = None)-> dict:
    """
    return the max and average of the absolute value of the nodal scalar product between functions f,g .
    The argument exclude can be used to exclude e.g. the boundary.
    """
    domain = f.function_space.mesh
    points = domain.geometry.x
    if exclude != None:
        points_to_exclude = np.invert(exclude(points.T))
        points = points[points_to_exclude]
    # print(points)
    yf = eval_continuous_function(f, points.T)
    yg = eval_continuous_function(g, points.T)
    res_vector = np.einsum('ij,ik->i', yf.T, yg.T) # / (np.linalg.norm(yf.T, axis=1, ord =2) * np.linalg.norm(yg.T, axis=1, ord =2) ) # scalar product
    return {"max": np.max(np.absolute(res_vector)), "avg": np.average(np.absolute(res_vector))}


def angle_between(f, g, dim):
    farray = f.x.array
    freshaped_array = np.reshape(    farray , (-1, dim))
    fnorms = np.linalg.norm(freshaped_array, axis=1, keepdims=True) 
    frescaled_array = freshaped_array / fnorms

    garray = g.x.array
    greshaped_array = np.reshape(    garray , (-1, dim))
    gnorms = np.linalg.norm(greshaped_array, axis=1, keepdims=True) 
    grescaled_array = greshaped_array / gnorms

    scalar_prod = np.clip(np.einsum('ij, ij->i', frescaled_array, grescaled_array), -1.0,1.0) # clipping along list scaled slightly down to prevent rounding error and nan or inf output
    return np.arccos(scalar_prod)

