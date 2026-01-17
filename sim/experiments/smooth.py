from argparse import Namespace
from functools import partial
import numpy as np
from dolfinx.mesh import create_rectangle, create_box, CellType, locate_entities_boundary
from dolfinx.cpp.mesh import DiagonalType
from dolfinx.fem import Constant
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from sim.common.meta_bcs import *
from sim.common.mesh import circumcenters

class smooth:
    def __init__(self, comm, args = Namespace()):
        # NAME
        self.name="Smooth solution"
        self.dim = args.dim
        self.dh = 2*int(args.dh/2)
        
        # MESH
        if self.dim == 3:
            self.mesh = create_box(comm, [np.array([-1, -1,-1]), np.array([1.0, 1.0,1.0])],  [self.dh,self.dh,self.dh], cell_type = CellType.tetrahedron)
            self.boundary = boundary_3d
        elif self.dim == 2:
            self.mesh = create_rectangle(comm, [np.array([-1.0, -1.0]), np.array([1.0, 1.0])],  [self.dh,self.dh], cell_type = CellType.triangle, diagonal=DiagonalType.left_right)
            self.boundary = boundary_2d

        # MESHTAGS
        # entities locate_entities_boundary(self.mesh, self.dim-1, self.boundary)
        self.meshtags = None

        #DG0 int points
        if args.mod in ["linear_dg"]:
            self.dg0_cells, self.dg0_int_points = circumcenters(self.mesh)

        # INIT FUNCTIONS WRT DIMENSION
        d0 = partial(get_d0, dim = self.dim)
        no_slip = partial(get_no_slip, dim = self.dim)

        # INITIAL CONDITIONS
        self.initial_conditions = {"v": no_slip, "p": (lambda x: np.full((x.shape[1],), 0.0)), "d": d0}

        # BOUNDARY CONDITIONS
        self.boundary_conditions = [
            meta_dirichletbc("v", "geometrical", no_slip,  marker = self.boundary),
            ]
        # self.boundary_conditions += [meta_dirichletbc("d", "geometrical", d0, marker = self.boundary)] # Comment out for homogeneous Neumann Boundary Conditions
    
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
