from argparse import Namespace
from functools import partial
import numpy as np
from dolfinx.mesh import create_rectangle, CellType, locate_entities_boundary
from dolfinx.cpp.mesh import DiagonalType
from sim.common.meta_bcs import *
from sim.common.mesh import circumcenters

"""
Experiment class to invoke a Laminar Flow using a Pressure Difference
"""

class poiseuille:
    def __init__(self, comm, args = Namespace()):
        # NAME
        self.name="Poiseuille flow"     
        self.dh = 2*int(args.dh/2)   
        self.dim = 2
        self.L = 2.0
        self.vel_profile = args.vel_profile
        self.bc_option = args.bc_option
        
        
        # MESH
        self.mesh = create_rectangle(comm, [np.array([0.0, -1.0]), np.array([self.L, 1.0])],  [self.dh,self.dh], cell_type = CellType.triangle, diagonal=DiagonalType.left_right)

        # MESHTAGS
        self.meshtags = None

        #DG0 int points
        if args.mod in ["linear_dg"]:
            self.dg0_cells, self.dg0_int_points = circumcenters(self.mesh)

        # INIT FUNCTIONS WRT DIMENSION
        if self.bc_option == 0:
            d0 = partial(d0_weak_flow, dim = self.dim)
        else:
            d0 = partial(d0_strong_flow, dim = self.dim)

        no_slip = partial(get_no_slip, dim = self.dim)

        # INITIAL CONDITIONS
        self.initial_conditions = {"v": no_slip, "p": (lambda x: np.full((x.shape[1],), 0.0)), "d": d0}
        
        # BOUNDARY CONDITIONS
        self.boundary_conditions =  [
                                    meta_dirichletbc("v", "geometrical", no_slip,  marker = self.top_or_bottom),   # no-slip velocity
                                    meta_dirichletbc("p", "geometrical", self.get_p0,  marker = self.left),        # pressure inlet
                                    meta_dirichletbc("p", "geometrical", self.get_p0,  marker = self.right),       # pressure outlet
                                    meta_dirichletbc("d", "geometrical", d0, marker = self.top_or_bottom),         # dirichlet for d
                                    ] 
        
    @property
    def info(self):
        return {"name":self.name}
    @property
    def has_exact_solution(self):
        return False

    def boundary(self, x: np.ndarray) -> np.ndarray:
        return np.logical_or.reduce((np.isclose(x[0], 0.0), np.isclose(x[0], self.L), np.isclose(x[1], -1.0), np.isclose(x[1], 1.0)))
    def top(self, x: np.ndarray) -> np.ndarray:
        return np.isclose(x[1], 1.0)
    def bottom(self, x: np.ndarray) -> np.ndarray:
        return np.isclose(x[1], -1.0)
    def left(self, x: np.ndarray) -> np.ndarray:
        return np.isclose(x[0], 0.0)
    def right(self, x: np.ndarray) -> np.ndarray:
        return np.isclose(x[0], self.L)
    def top_or_bottom(self, x: np.ndarray) -> np.ndarray:
        return np.logical_or.reduce((np.isclose(x[1], -1.0), np.isclose(x[1], 1.0)))
    
    def get_p0(self, x: np.ndarray) -> np.ndarray:
        # x hase shape (dimension, points)
        values = np.zeros((x.shape[1], )) # values is going to be the output

        pressure_difference = self.vel_profile

        values = pressure_difference  - x[0]/self.L*pressure_difference
        
        return values

    
def d0_weak_flow(x: np.ndarray, dim: int)-> np.ndarray:
    if dim not in [2,3]: raise ValueError("Dimension "+str(dim)+" not supported.")
    # x hase shape (dimension, points)
    values = np.zeros((dim, x.shape[1])) # values is going to be the output

    values[0]= 0
    values[1]= 1
    if dim == 3: values[2]=0.0

    return values

def d0_strong_flow(x: np.ndarray, dim: int)-> np.ndarray:
    if dim not in [2,3]: raise ValueError("Dimension "+str(dim)+" not supported.")
    # x hase shape (dimension, points)
    values = np.zeros((dim, x.shape[1])) # values is going to be the output
    dofs = x[1] >= 0 
    # dofs = ~np.isclose(x[1], -1.0)

    values[0][dofs]= 0
    values[1][dofs]= 1

    values[0][~dofs]= 0
    values[1][~dofs]= -1
    if dim == 3: values[2]=0.0

    return values
    
def get_no_slip(x: np.ndarray, dim: int) -> np.ndarray:
    # x hase shape (dimension, points)
    if dim >1:
        values = np.zeros((dim, x.shape[1]))
    else: values = np.zeros(x.shape[1])
    return values
