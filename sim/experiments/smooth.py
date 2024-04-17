from argparse import Namespace
from functools import partial
import numpy as np
from dolfinx.mesh import create_rectangle, create_box, CellType
from dolfinx.fem import Constant
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from sim.experiments.bcs_wo_fs import *
from sim.common.common_methods import set_attributes

class smooth:
    def __init__(self, args = Namespace()):
        self.name="Smooth solution"
        # - model parameters namely v_el, const_A, nu, mu_1, mu_4, mu_5, mu_6, lam
        default_attributes = {"dh":2**4, "dt":0.0005, "T":2.0, "dim": 2, "t0": 0.0, "v_el":1.0, "const_A":1.0, "nu":0.1,"mu_1":1.0, "mu_4": 0.1, "mu_5":1.0, "mu_6":1.0 , "lam":1.0}
        set_attributes(self, default_attributes, args)

        if self.dim == 3:
            self.mesh = create_box(MPI.COMM_WORLD, [np.array([-1, -1,-1]), np.array([1.0, 1.0,1.0])],  [self.dh,self.dh,self.dh], cell_type = CellType.tetrahedron)
            self.boundary = boundary_3d
        elif self.dim == 2:
            self.mesh = create_rectangle(MPI.COMM_WORLD, [np.array([-1.0, -1.0]), np.array([1.0, 1.0])],  [self.dh,self.dh], cell_type = CellType.triangle)
            self.boundary = boundary_2d
        self.meshtags = None
        d0 = partial(get_d0, dim = self.dim)
        no_slip = partial(get_no_slip, dim = self.dim)
         # - initial conditions
        self.initial_conditions = {"v": no_slip, "p": (lambda x: np.full((x.shape[1],), 0.0)), "d": d0}
        # boundary conditions: no-slip for v and homogeneous neumann BC for d
        self.boundary_conditions = [dirichlet_bc_wo_fs("v", "geometrical", no_slip,  marker = self.boundary)] #, \
                                    # dirichlet_bc_wo_fs("d", "geometrical", d0, marker = self.boundary)]
    
    @property
    def info(self):
        return {"name":self.name}
    @property
    def has_exact_solution(self):
        return False
    
def boundary_3d(x: np.ndarray) -> np.ndarray:
    return np.logical_or.reduce((np.isclose(x[0], -1.0), np.isclose(x[0], 1.0), np.isclose(x[1], -1.0), np.isclose(x[1], 1.0),np.isclose(x[2], -1.0), np.isclose(x[2], 1.0)))

def boundary_2d(x: np.ndarray) -> np.ndarray:
    return np.logical_or.reduce((np.isclose(x[0], -1.0), np.isclose(x[0], 1.0), np.isclose(x[1], -1.0), np.isclose(x[1], 1.0)))

def get_d0(x: np.ndarray, dim: int)-> np.ndarray:
    if dim not in [2,3]: raise ValueError("Dimension "+str(dim)+" not supported.")
    # x hase shape (dimension, points)
    values = np.zeros((dim, x.shape[1])) # values is going to be the output
    
    # Setting defects
    values[0]= np.sin( 2.0*np.pi*(np.cos(x[0])-np.sin(x[1]) ) )
    values[1]= np.cos( 2.0*np.pi*(np.cos(x[0])-np.sin(x[1]) ) )
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
