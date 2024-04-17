import warnings
from typing import Union
from pathlib import Path
from mpi4py import MPI
import numpy as np
from dolfinx.fem import Function, functionspace, form, VectorFunctionSpace, assemble_scalar, Expression, ElementMetaData
from dolfinx.mesh import GhostMode
from ufl import dx, grad, inner
from dolfinx.io import XDMFFile
from ufl.core.expr import Expr
from ufl import div, dx, grad, inner, VectorElement, FiniteElement, MixedElement, TrialFunction, TrialFunctions, TestFunctions, TestFunction, split, Measure, lhs, rhs, FacetNormal

def errornorm(uh: Function, u_ex: Function, norm: str = "L2", degree_raise: int =3, mode: str = "abs")-> float:
    """
    Computes the errornom of two functions.
    Options are:
        norm = "L2", "H10", "H1", "inf"
        mode = "
    adapted from: https://jsdokken.com/dolfinx-tutorial/chapter4/convergence.html (12.06.2023)
    """
    # if our exact solution is already a FEM function we take the latter as the right function space
    # if u_ex is a callable, we raise the FEM space of the first entry by degree_raise to get a better approximation
    if type(u_ex) == Function:
        FS = u_ex.function_space
    else:
        FS = uh.function_space
    
    # Create higher order function space
    dim = FS.num_sub_spaces
    degree = FS.ufl_element().degree()
    family = FS.ufl_element().family()
    mesh = FS.mesh
    
    if degree_raise > 0:   
        W = functionspace(mesh, ElementMetaData(family, degree+ degree_raise, shape = (dim,))) #v0.7     
        # if dim >1:
        #     # W = VectorFunctionSpace(mesh, (family, degree+ degree_raise), dim = dim) # v0.6
        # else: 
        #     W = functionspace(mesh, (family, degree+ degree_raise))
    else:
        W = FS
    # W is the Function space in which the error will be computed
    # Interpolate approximate solution
    if type(u_ex) == Function or degree_raise > 0:
        u_W = Function(W)
        u_W.interpolate(uh)
    else:
        u_W = uh

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    
    if type(u_ex) == Function:
        u_ex_W = u_ex
    else:
        u_ex_W = Function(W)
        if isinstance(u_ex, Expr):
            print(isinstance(u_ex, Expr))
            u_expr = Expression(u_ex, W.element.interpolation_points())
            u_ex_W.interpolate(u_expr)
        else:
            u_ex_W.interpolate(u_ex)
    
    # Compute the error in the higher order function space
    e_W = Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array
    
    # Integrate the error
    if norm == "L2":
        error = form(inner(e_W, e_W) * dx)
        error_local = assemble_scalar(error)
        error_global = np.sqrt(mesh.comm.allreduce(error_local, op=MPI.SUM))
        if mode == "rel": 
            divisor = form(inner(u_ex_W, u_ex_W) * dx)
            divisor_local = assemble_scalar(divisor)
            divisor_global = np.sqrt(mesh.comm.allreduce(divisor_local, op=MPI.SUM))
            error_global = error_global / divisor_global
    elif norm == "H10":
        error = form(inner(grad(e_W), grad(e_W)) * dx)
        error_local = assemble_scalar(error)
        error_global = np.sqrt(mesh.comm.allreduce(error_local, op=MPI.SUM))
        if mode == "rel": 
            divisor = form(inner(grad(u_ex_W), grad(u_ex_W)) * dx)
            divisor_local = assemble_scalar(divisor)
            divisor_global = np.sqrt(mesh.comm.allreduce(divisor_local, op=MPI.SUM))
            error_global = error_global / divisor_global
    elif norm == "H1":
        error = form(inner(grad(e_W), grad(e_W)) * dx + inner(e_W, e_W) * dx)
        error_local = assemble_scalar(error)
        error_global = np.sqrt(mesh.comm.allreduce(error_local, op=MPI.SUM))
        if mode == "rel": 
            divisor = form(inner(grad(u_ex_W), grad(u_ex_W)) * dx + inner(u_ex_W, u_ex_W) * dx)
            divisor_local = assemble_scalar(divisor)
            divisor_global = np.sqrt(mesh.comm.allreduce(divisor_local, op=MPI.SUM))
            error_global = error_global / divisor_global
    elif norm == "inf":
        error_global = np.linalg.norm(e_W.x.array[:], np.inf)
        if mode == "rel": 
            error_global = error_global /np.linalg.norm(u_ex_W.x.array[:], np.inf) 
    return error_global 

class TemporalFunctionReader:
    def __init__(self, filepath: Path, elm: ElementMetaData ):
        """
        elm: ElementMetaData(family, order, shape = (dim,))
        """
        try:
            import adios4dolfinx
        except ModuleNotFoundError:
            raise ModuleNotFoundError("To use the class TemporalFunctionReader, you need the Module adios4dolfinx.")
        
        self.path       = filepath
        self.element    = elm
        self.domain     = adios4dolfinx.read_mesh(MPI.COMM_WORLD, self.path, engine = "BP4", ghost_mode = GhostMode.shared_facet)
        self.FS         = functionspace(self.domain, self.element)
        self.u          = Function(self.FS)

    def read_function(self, time: float = 0.0)-> Function:
        try:
            import adios4dolfinx
        except ModuleNotFoundError:
            raise ModuleNotFoundError("To use the class TemporalFunctionReader and its method, you need the Module adios4dolfinx.")
        adios4dolfinx.read_function(self.u, self.path, engine = "BP4", time=time)
        return self.u

def errornorm_bochner(u_h: TemporalFunctionReader, u_ex: TemporalFunctionReader, time_points: np.ndarray, norm: str = "L2", ord: Union[float, np.inf, None] = None, mode: str = "abs")-> float | list:   
    """
    Computes a Bochner-Norm of the difference of two functions in the same FEM space, i.e.
        uh:             an approximate solution, and
        u_ex:           the reference solution,
        time_points:    np.ndarray to describe the time points. Should start with 0.0.

    Options:
        norm:   spatial norm, same options as for errornorm
        ord:    temporal norm order for a Bochner-Norm, for None arrays of time points and errors are returned
        mode:   "abs" for absolute error, "rel" for relative error
    """ 
    try:
        import adios4dolfinx
    except ModuleNotFoundError:
        raise ModuleNotFoundError("To use the method errornorm_bochner, you need the Module adios4dolfinx.")
    
    
    
    errors = []
    times =[]
    for t in time_points:

        try:
            u_ex = adios4dolfinx.read_function(time=t)
            u_h = adios4dolfinx.read_function(time=t)
            err = errornorm(u_h, u_ex, norm = norm, degree_raise = 0, mode = mode)
            errors.append(err)
            times.append(t)
        except KeyError as e:
            warnings.warn("Encountered Key Error: "+str(e)+". Skipping the time step "+str(t))
    
    if ord == None:
        return [np.array(times), np.array(errors)]
    else:
        return array_to_bochner( np.array(errors), np.array(times), ord = ord)

def array_to_bochner(values: np.ndarray, time_points: np.ndarray, ord: Union[float, np.inf] = 2):
    """
    computes the temporal error based on an array of spatial errors on a partition of time
    """
    if time_points[0] != 0.0:
        time_points = np.array([0.0, time_points])
    
    weights = np.diff(time_points)

    if ord == np.inf:
        return np.linalg.norm(values, np.inf)
    else:
        return np.inner(weights, values**ord)**(1/ord)