import os
from argparse import Namespace
from functools import partial
import numpy as np
from basix.ufl import element
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary
from sim.common.meta_bcs import *
from sim.common.common_fem_methods import nodal_normalization
from sim.common.error_computation import *
from sim.common.mesh import circumcenters

"""
class for a standard benchmark setting for the ericksen-leslie model: 
    annihilation of two defects without an initial flow 
"""

class spiral:
    def __init__(self, comm, args = Namespace()):
        # NAME
        self.name="spiral"

        # SECTION - MESH AND MESHTAGS
        mesh_loc = "input/meshes/spiral2D_"+str(args.dh)
        
        if os.path.isfile(mesh_loc+".xdmf"):
            # mesh exists in xdmf format
            with XDMFFile(comm, mesh_loc+".xdmf" , "r") as f:
                self.mesh = f.read_mesh()
                self.mesh.topology.create_connectivity(self.mesh.topology.dim - 1, self.mesh.topology.dim)
                self.meshtags = f.read_meshtags(self.mesh, name =  "mesh_tags")

        else:
            raise FileNotFoundError("Could not find any mesh in msh or xdmf format under "+mesh_loc+"... To run this experiment the according mesh is needed as input.")
        
        self.dim = 2
        self.boundary = boundary
        inside, outside  =  self.meshtags.find(2) ,  self.meshtags.find(3)

        #DG0 int points
        if args.mod in ["linear_dg"]:
            self.dg0_cells, self.dg0_int_points = circumcenters(self.mesh)
        #!SECTION


        # INIT FUNCTIONS WRT DIMENSION
        d0 = partial(get_d0, dim = self.dim)
        dbc = partial(get_d_bc, dim = self.dim)
        v0 = partial(get_no_slip, dim = self.dim)

        # INITIAL CONDITIONS
        self.initial_conditions = {"v": v0, "p": (lambda x: np.full((x.shape[1],), 0.0)), "d": d0}

        # BOUNDARY CONDITIONS
        self.boundary_conditions = [meta_dirichletbc("d", "topological", dbc, entities = outside, marker = bd_outside, meshtag=3), \
                                    meta_dirichletbc("d", "topological", dbc, entities = inside, marker = bd_inside, meshtag=2), \
                                    meta_dirichletbc("v", "topological", v0, entities= outside, marker = bd_outside, meshtag=3), \
                                    meta_dirichletbc("v", "topological", v0, entities= inside, marker = bd_inside, meshtag=2)]
    @property
    def info(self):
        return {"name":self.name}
    @property
    def has_exact_solution(self):
        if self.dim == 2: return True
        else: return False
    
    def exact_sol(self, x: np.ndarray) -> np.ndarray:
    # x hase shape (dimension, points)
        values = np.zeros((self.dim, x.shape[1]))
        r = np.sqrt(x[0]**2 + x[1]**2)    
        r0 = 1
        r1 = 2
        angle = np.pi/2 * np.log(r/r0)/np.log(r1/r0)
        # Rotation matrix: The matrix rotates points clockwise around the origin by an angle θ
        # R(θ) =    | cos(θ)  sin(θ) |
        #           | -sin(θ)   cos(θ) |

        # Rotate radial direction
        values[0]= np.cos(angle)*x[0] + np.sin(angle)*x[1]
        values[1]= (-1)*np.sin(angle)*x[0] + np.cos(angle)*x[1]

        # renormalization
        norms = np.linalg.norm(values, ord = 2, axis = 0) # compute euclidean norm
        values = values / norms # renormalize

        return values
    
    def compute_error(self, comm, uh, time, norm = "L2", degree_raise = 3, family = None, degree = None):
        # arg time is not necessary for this class since we only consider the steady state limit of phi as exact solution
        # obtain FunctionSpace from approximate function
        if degree == None:
            degree = uh.function_space.ufl_element().degree
        if family == None:
            family = uh.function_space.ufl_element().family_name
        
        # raise VectorFunctionSpace
        FSr = functionspace(self.mesh, element(family, self.mesh.basix_cell(), degree+degree_raise, shape=(self.dim,)))
        uex = Function(FSr)
        uex.interpolate(self.exact_sol)
        uex.x.scatter_forward() 

        # Normalize uh to only compare the angle essentially:
        nodal_normalization(uh,2)

        err_local   = assemble_scalar(form(inner(uh-uex,uh-uex)*dx))
        err         = comm.allreduce(err_local, op=MPI.SUM)**0.5

        return err

#SECTION - CUSTOM FUNCTIONS

def get_d0(x: np.ndarray, dim: int) -> np.ndarray:
    if dim not in [2,3]: raise ValueError("Dimension "+str(dim)+" not supported.")
    # x hase shape (dimension, points)

    values = np.zeros((dim, x.shape[1])) # values is going to be the output
    outside_dofs = bd_outside(x) # array of True and False giving the defect locations  

    # Setting 
    # - normal to the boundary with some tilt described by eta
    values[0]=x[0] 
    values[1]=x[1] 
    if dim == 3: values[2]=0.0

    # Setting outside BC    
    # - tangential to sphere 
    values[0][outside_dofs]=x[1][outside_dofs]
    values[1][outside_dofs]=-x[0][outside_dofs] 
    if dim == 3: values[2][outside_dofs]=0.0

    # renormalization
    norms = np.linalg.norm(values, ord = 2, axis = 0) # compute euclidean norm
    values = values / norms # renormalize
    return values

def get_d_bc(x: np.ndarray, dim: int) -> np.ndarray:
    if dim not in [2,3]: raise ValueError("Dimension "+str(dim)+" not supported.")
    # x hase shape (dimension, points)

    values = np.zeros((dim, x.shape[1])) # values is going to be the output  

    # Setting 
    # - normal to the boundary with some tilt described by eta
    r = (x[0]**2 + x[1]**2)**(1/2)
    inner_half = r < 1.5
    outer_half = r >= 1.5
    values[0][outer_half]= x[1][outer_half] 
    values[1][outer_half]= -x[0][outer_half]
    values[0][inner_half]= x[0][inner_half]
    values[1][inner_half]= x[1][inner_half]
    if dim == 3: values[2]=0.0

    # renormalization
    norms = np.linalg.norm(values, ord = 2, axis = 0) # compute euclidean norm
    values = values / norms # renormalize
    return values


def get_no_slip(x: np.ndarray, dim: int) -> np.ndarray:
    # x hase shape (dimension, points)
    if dim >1:
        values = np.zeros((dim, x.shape[1]))
    else: values = np.zeros(x.shape[1])
    return values

def unit_radials(x: np.ndarray, dim: int) -> np.ndarray:
    if dim not in [2,3]: raise ValueError("Dimension "+str(dim)+" not supported.")
    # x hase shape (dimension, points)

    values = np.zeros((dim, x.shape[1])) # values is going to be the output 

    values[0]=x[0] 
    values[1]=x[1] 
    if dim == 3: values[2]=0.0

    # renormalization
    norms = np.linalg.norm(values, ord = 2, axis = 0) # compute euclidean norm
    values = values / norms # renormalize
    return values
#!SECTION

#SECTION - GEOMETRIC BOUNDARY DESCRIPTION

def boundary(x: np.ndarray) -> np.ndarray:
    return np.logical_or(bd_inside(x), bd_outside(x))

def bd_outside(x):
    return np.isclose(x[0]**2 + x[1]**2, 4)

def bd_inside(x):
    return np.isclose(x[0]**2 + x[1]**2, 1, atol=.1)

#!SECTION
