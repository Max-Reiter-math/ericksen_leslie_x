from argparse import Namespace
from functools import partial
import numpy as np
from dolfinx.mesh import create_rectangle, create_box, CellType, locate_entities, locate_entities_boundary, meshtags, meshtags_from_entities
from dolfinx.cpp.mesh import DiagonalType
from dolfinx.fem import Constant
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from sim.common.meta_bcs import *
from sim.common.mesh import circumcenters, midpoints

"""
class for a standard benchmark setting for the ericksen-leslie model: 
    annihilation of two defects without an initial flow 
"""

class annihilation_2:
    def __init__(self, comm, args = Namespace()):
        # NAME
        self.name="annihilation of two defects"
        self.dim = args.dim
        self.dh = args.dh

        # MESH
        if args.mod in ["linear_dg"] and args.dim == 3:
            celltype = CellType.hexahedron
        elif args.dim == 3:
            celltype = CellType.tetrahedron
        elif args.dim == 2:
            celltype = CellType.triangle

        if self.dim == 3:
            self.mesh = create_box(MPI.COMM_WORLD, [np.array([-0.5, -0.5,-0.5]), np.array([0.5, 0.5,0.5])],  [self.dh,self.dh,self.dh], cell_type = celltype)
            self.boundary = boundary_3d
        elif self.dim == 2:
            self.mesh = create_rectangle(MPI.COMM_WORLD, [np.array([-0.5, -0.5]), np.array([0.5, 0.5])],  [self.dh,self.dh], cell_type = celltype, diagonal = DiagonalType.left_right)
            self.boundary = boundary_2d
        
        # MESHTAGS
        # entities        = locate_entities_boundary(self.mesh, self.dim-1, self.boundary)
        self.meshtags   = None # meshtags(self.mesh, self.dim-1, entities, 0) # mark the full boundary with the marker 0

        #DG0 int points
        if args.mod in ["linear_dg"]:
            if self.dim == 2:
                self.dg0_cells, self.dg0_int_points = circumcenters(self.mesh)
            elif self.dim == 3:
                self.dg0_cells, self.dg0_int_points = midpoints(self.mesh)

        # INIT FUNCTIONS WRT DIMENSION
        d0 = partial(get_d0, dim = self.dim, dh=self.dh)        
        dbc = partial(get_dbc, dim= self.dim, dh=self.dh)
        no_slip = partial(get_no_slip, dim = self.dim)

        # INITIAL CONDITIONS
        self.initial_conditions = {"v": no_slip, "p": (lambda x: np.full((x.shape[1],), 0.0)), "d": d0}

        # BOUNDARY CONDITIONS
        self.boundary_conditions = [
                                    meta_dirichletbc("v", "geometrical", no_slip,  marker = self.boundary), 
                                    meta_dirichletbc("d", "geometrical", dbc, marker = boundary_x),
                                    meta_dirichletbc("d", "geometrical", dbc, marker = boundary_y),
                                    ]
        if self.dim == 3:
            self.boundary_conditions += [meta_dirichletbc("d", "geometrical", dbc, marker = boundary_z),]
    
    @property
    def info(self):
        return {"name":self.name}
    @property
    def has_exact_solution(self):
        return False
    
def boundary_x(x: np.ndarray) -> np.ndarray:
    return np.logical_or(np.isclose(x[0], -0.5), np.isclose(x[0], 0.5))

def boundary_y(x: np.ndarray) -> np.ndarray:
    return np.logical_or(np.isclose(x[1], -0.5), np.isclose(x[1], 0.5))

def boundary_z(x: np.ndarray) -> np.ndarray:
    return np.logical_or(np.isclose(x[2], -0.5), np.isclose(x[2], 0.5))

def boundary_3d(x: np.ndarray) -> np.ndarray:
    return np.logical_or.reduce((np.isclose(x[0], -0.5), np.isclose(x[0], 0.5), np.isclose(x[1], -0.5), np.isclose(x[1], 0.5),np.isclose(x[2], -0.5), np.isclose(x[2], 0.5)))

def boundary_2d(x: np.ndarray) -> np.ndarray:
    return np.logical_or.reduce((np.isclose(x[0], -0.5), np.isclose(x[0], 0.5), np.isclose(x[1], -0.5), np.isclose(x[1], 0.5)))

def get_d0(x: np.ndarray, dim: int, dh: int)-> np.ndarray:
    if dim not in [2,3]: raise ValueError("Dimension "+str(dim)+" not supported.")
    # x hase shape (dimension, points)
    values = np.zeros((dim, x.shape[1])) # values is going to be the output

    #NOTE - the following makes mostly sense for the DG0 case
    tol = 0.9*1/dh
    order = np.inf
    # array of True and False giving the defect locations
    defects1 = np.linalg.norm(np.stack((x[0]-0.25,x[1])), ord = order, axis = 0) <= tol
    defects2 = np.linalg.norm(np.stack((x[0]+0.25,x[1])), ord = order, axis = 0) <= tol
    defects = np.logical_or(defects1,defects2)
    no_defect = np.invert(defects)
    
    # Setting defects
    values[0][defects]=0.0
    values[1][defects]=0.0
    if dim == 3: values[2][defects]=1.0

    # Setting the rest
    values[0][no_defect]=4.0*(x[0][no_defect]**2)+4*(x[1][no_defect]**2)-0.25
    values[1][no_defect]=2.0*x[1][no_defect]
    if dim == 3: values[2][no_defect]=0.0

    # renormalization
    norms = np.linalg.norm(values, ord = 2, axis = 0) # compute euclidean norm
    values = values / norms # renormalize
    return values

def get_dbc(x: np.ndarray, dim: int, dh: int)-> np.ndarray:
    if dim not in [2,3]: raise ValueError("Dimension "+str(dim)+" not supported.")
    # x hase shape (dimension, points)
    
    #NOTE - for the DG case we move the interpolation points close enough to the boundary onto the boundary
    h=1/dh
    close_to_boundary_dg_dofs = [np.logical_or(x[i] == np.max(x[i]), x[i] == np.min(x[i])) for i in range(dim)]
    for i in range(dim):
        x[i][close_to_boundary_dg_dofs[i]] = 0.5 * x[i][close_to_boundary_dg_dofs[i]] / np.abs(x[i][close_to_boundary_dg_dofs[i]])
    
    # now reapplying the previous function
    values = get_d0(x, dim, dh)
    return values
     
def get_no_slip(x: np.ndarray, dim: int) -> np.ndarray:
    # x hase shape (dimension, points)
    if dim >1:
        values = np.zeros((dim, x.shape[1]))
    else: values = np.zeros(x.shape[1])
    return values
