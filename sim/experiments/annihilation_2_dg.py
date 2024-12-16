from argparse import Namespace
from functools import partial
import numpy as np
from dolfinx.mesh import create_rectangle, create_box, CellType, locate_entities, locate_entities_boundary, meshtags, meshtags_from_entities
from dolfinx.fem import Constant
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from sim.common.meta_bcs import *
from sim.common.common_methods import set_attributes

"""
class for a standard benchmark setting for the ericksen-leslie model: 
    annihilation of two defects without an initial flow 
"""

class annihilation_2_dg:
    def __init__(self, args = Namespace()):
        # NAME
        self.name="annihilation of two defects"
        # PARAMETERS: v_el, const_A, nu, mu_1, mu_4, mu_5, mu_6, lam
        default_attributes = {"dim" : 3, "dh" : 2**4, "dt" : 0.0005, "T" : 0.1, "t0":0.0, "v_el":.25, "const_A":1.0, "nu":1.0,"mu_1":1.0, "mu_4": 1.0, "mu_5":1.0, "mu_6":1.0 , "lam":1.0}
        set_attributes(self, default_attributes, args)

        # MESH
        if self.dim == 3:
            self.mesh = create_box(MPI.COMM_WORLD, [np.array([-0.5, -0.5,-0.5]), np.array([0.5, 0.5,0.5])],  [self.dh,self.dh,self.dh], cell_type = CellType.tetrahedron)
            self.boundary = boundary_3d
        elif self.dim == 2:
            self.mesh = create_rectangle(MPI.COMM_WORLD, [np.array([-0.5, -0.5]), np.array([0.5, 0.5])],  [self.dh,self.dh], cell_type = CellType.triangle)
            self.boundary = boundary_2d
        
        # MESHTAGS
        # entities        = locate_entities_boundary(self.mesh, self.dim-1, self.boundary)
        self.meshtags   = None # meshtags(self.mesh, self.dim-1, entities, 0) # mark the full boundary with the marker 0

        # INIT FUNCTIONS WRT DIMENSION
        d0 = partial(get_d0, dim = self.dim, dh=self.dh)
        dbc = partial(get_dbc, dim= self.dim)
        no_slip = partial(get_no_slip, dim = self.dim)

        # INITIAL CONDITIONS
        self.initial_conditions = {"v": no_slip, "p": (lambda x: np.full((x.shape[1],), 0.0)), "d": d0}

        # BOUNDARY CONDITIONS
        self.boundary_conditions = [meta_dirichletbc("v", "geometrical", no_slip,  marker = self.boundary), \
                                    meta_dirichletbc("d", "geometrical", dbc, marker = self.boundary) ]
    
    @property
    def info(self):
        return {"name":self.name}
    @property
    def has_exact_solution(self):
        return False
    
def boundary_3d(x: np.ndarray) -> np.ndarray:
    return np.logical_or.reduce((np.isclose(x[0], -0.5), np.isclose(x[0], 0.5), np.isclose(x[1], -0.5), np.isclose(x[1], 0.5),np.isclose(x[2], -0.5), np.isclose(x[2], 0.5)))

def boundary_2d(x: np.ndarray) -> np.ndarray:
    return np.logical_or.reduce((np.isclose(x[0], -0.5), np.isclose(x[0], 0.5), np.isclose(x[1], -0.5), np.isclose(x[1], 0.5)))

def get_d0(x: np.ndarray, dim: int, dh: int)-> np.ndarray:
    if dim not in [2,3]: raise ValueError("Dimension "+str(dim)+" not supported.")
    # x hase shape (dimension, points)
    values = np.zeros((dim, x.shape[1])) # values is going to be the output

    #NOTE - the following makes mostly sense for the DG0 case
    tol = 1/dh 
    # array of True and False giving the defect locations
    defects = np.logical_and(np.isclose(x[1],0.0, atol=tol),  np.logical_or( np.isclose(x[0],0.25, atol=tol) , np.isclose(x[0],-0.25, atol=tol)) )  # infty norm for localization
    # defects = np.logical_or(np.isclose(np.abs(x[1])+np.abs(x[0]+0.25),0.0,atol=tol),np.isclose(np.abs(x[1])+np.abs(x[0]-0.25),0.0,atol=tol)) # 1- norm for localization
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

def get_dbc(x: np.ndarray, dim: int)-> np.ndarray:
    if dim not in [2,3]: raise ValueError("Dimension "+str(dim)+" not supported.")
    # x hase shape (dimension, points)
    values = np.zeros((dim, x.shape[1])) # values is going to be the output

    # Setting the rest
    values[0]=4.0*(x[0]**2)+4*(x[1]**2)-0.25
    values[1]=2.0*x[1]
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
