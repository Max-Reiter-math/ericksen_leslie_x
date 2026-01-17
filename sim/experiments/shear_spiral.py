import os
from argparse import Namespace
from functools import partial
import numpy as np
from dolfinx.io import XDMFFile
from sim.common.meta_bcs import *
from sim.common.mesh import circumcenters

"""
Class for a shear flow simulation in a spiral
"""

class shear_spiral:
    def __init__(self, comm, args = Namespace()):
        # NAME
        self.name="Shear flow spiral"

        self.dim = 2
        self.vel_profile = args.vel_profile

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
        
        self.boundary = boundary
        inside, outside  =  self.meshtags.find(2) ,  self.meshtags.find(3)

        #DG0 int points
        if args.mod in ["linear_dg"]:
            self.dg0_cells, self.dg0_int_points = circumcenters(self.mesh)
        #!SECTION

        # INITIAL CONDITIONS
        self.initial_conditions = {"v": no_slip, "p": (lambda x: np.full((x.shape[1],), 0.0)), "d": d_0}

        # BOUNDARY CONDITIONS
        self.boundary_conditions = [meta_dirichletbc("d", "topological", d_0, entities = outside, marker = bd_outside, meshtag=3), \
                                    meta_dirichletbc("d", "topological", d_0, entities = inside, marker = bd_inside, meshtag=2), \
                                    meta_dirichletbc("v", "topological", no_slip, entities= outside, marker = bd_outside, meshtag=3), \
                                    meta_dirichletbc("v", "topological", self.v_bc, entities= inside, marker = bd_inside, meshtag=2)]
    @property
    def info(self):
        return {"name":self.name}
    @property
    def has_exact_solution(self):
        return False

    def v_bc(self, x: np.ndarray) -> np.ndarray:

        values = np.zeros((2, x.shape[1])) # values is going to be the output
    
        # - tangential to sphere 
        values[0]=x[1]
        values[1]=-x[0] 

        # rescaling
        U = self.vel_profile
        radius = np.linalg.norm(values, ord = 2, axis = 0) # compute euclidean norm
        values = U * values / radius *( 2 - radius ) # renormalize
        return values

def d_0(x: np.ndarray) -> np.ndarray:

    values = np.zeros((2, x.shape[1])) # values is going to be the output
  
    # - tangential to sphere 
    values[0]=x[1]
    values[1]=-x[0] 

    # renormalization
    norms = np.linalg.norm(values, ord = 2, axis = 0) # compute euclidean norm
    values = values / norms # renormalize
    return values



def no_slip(x: np.ndarray) -> np.ndarray:
    # x hase shape (dimension, points)
    values = np.zeros((2, x.shape[1]))
    return values

def boundary(x: np.ndarray) -> np.ndarray:
    return np.logical_or(bd_inside(x), bd_outside(x))

def bd_outside(x):
    return np.isclose(x[0]**2 + x[1]**2, 4)

def bd_inside(x):
    return np.isclose(x[0]**2 + x[1]**2, 1, atol=.1)

