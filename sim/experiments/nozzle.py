import os
from argparse import Namespace
from functools import partial
import numpy as np
from dolfinx.mesh import create_rectangle, create_box, CellType, locate_entities_boundary, GhostMode
from dolfinx.fem import Constant
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from sim.common.meta_bcs import *

from sim.common.mesh import circumcenters

"""
Poiseuille (Pressure-Driven) Flow in a Trapez.
Usage examples:
    python -m sim.run -m linear_cg -e nozzle -dh 10 -vtx 1 -sid nozzle5 -fsr 0.0 -mu1 -0.155 -mu5 0.7324 -mu6 -0.394 -lam 1.2085 -gamma 1.083 -beta 1.0 -K1 0.5 -dt 0.1 -T 3.0 -vel_profile 3.0
    python -m sim.run -m linear_cg -e nozzle -dh 10 -vtx 1 -sid nozzle -fsr 0.0 -mu1 -0.155 -mu5 0.7324 -mu6 -0.394 -lam 1.2085 -gamma 1.083 -beta 1.0 -K1 0.5 -dt 0.1 -T 3.0 -vel_profile 3.0
    python -m sim.run -m linear_cg -e nozzle -dh 10 -vtx 1 -sid nozzle3 -fsr 0.0 -mu1 -0.155 -mu5 0.7324 -mu6 -0.394 -lam 1.2085 -gamma 1.083 -beta 1.0 -K1 0.5 -dt 0.1 -T 3.0 -vel_profile 10.0
"""

class nozzle:
    def __init__(self, comm, args = Namespace()):
        # NAME
        self.name="Nozzle Flow"

        self.dim = 2
        self.W = 2.0 # nozzle diameter at top
        self.w = 1.5 # nozzle diameter at bottom
        self.h = 1.0 # nozzle height
        self.vel_profile = args.vel_profile

        # SECTION - MESH AND MESHTAGS
        mesh_loc = "input/meshes/nozzle_"+str(args.dh)
        
        if os.path.isfile(mesh_loc+".xdmf"):
            # mesh exists in xdmf format
            with XDMFFile(comm, mesh_loc+".xdmf" , "r") as f:
                self.mesh = f.read_mesh()
                self.mesh.topology.create_connectivity(self.mesh.topology.dim - 1, self.mesh.topology.dim)
                self.meshtags = f.read_meshtags(self.mesh, name =  "mesh_tags")

        else:
            raise FileNotFoundError("Could not find any mesh in msh or xdmf format under "+mesh_loc+"... To run this experiment the according mesh is needed as input.")
        
        inlet, outlet, walls  =  self.meshtags.find(2) ,  self.meshtags.find(3), self.meshtags.find(4)

        #DG0 int points
        if args.mod in ["linear_dg"]:
            self.dg0_cells, self.dg0_int_points = circumcenters(self.mesh)
        #!SECTION

        # INITIAL CONDITIONS
        self.initial_conditions = {"v": no_slip, "p": (lambda x: np.full((x.shape[1],), 0.0)), "d": d_0}

        # BOUNDARY CONDITIONS
        self.boundary_conditions = [meta_dirichletbc("d", "topological", self.d_bc, entities = walls, meshtag=4), 
                                    meta_dirichletbc("v", "topological", no_slip, entities= walls, meshtag=4),      # no slip
                                    meta_dirichletbc("p", "topological", (lambda x: np.full((x.shape[1],), 0.0)),  entities= outlet, meshtag=3),   # pressure
                                    meta_dirichletbc("p", "topological", (lambda x: np.full((x.shape[1],), self.vel_profile)),  entities= inlet, meshtag=2),   # pressure
                                    ]
        
    @property
    def info(self):
        return {"name":self.name}
    @property
    def has_exact_solution(self):
        return False
    
    def d_bc(self, x: np.ndarray)-> np.ndarray:
        
        # x hase shape (dimension, points)
        values = np.zeros((2, x.shape[1])) # values is going to be the output

        left = x[0] < 0
        right = x[0] > 0

        # planar boundary conditions
        # values[0][left] = (self.W-self.w)/2
        # values[0][right] = (self.w-self.W)/2
        # values[1] = (-1)*self.h

        # homeotropic boundary conditions
        values[0][left] = self.h
        values[0][right] = self.h
        values[1][left] = (self.W-self.w)/2
        values[1][right] = (self.w-self.W)/2

        # renormalization
        norms = np.linalg.norm(values, ord = 2, axis = 0) # compute euclidean norm
        values = values / norms # renormalize
        return values
    
def d_0(x: np.ndarray)-> np.ndarray:
    # planar boundary conditions
    # x hase shape (dimension, points)
    values = np.zeros((2, x.shape[1])) # values is going to be the output

    np.random.seed(20250909)

    angles = np.pi + np.random.rand(x.shape[1])*np.pi # this way all directors are only in a 180 degree radius

    values[0] = np.cos(angles)
    values[1] = np.sin(angles)

    # renormalization 
    # norms = np.linalg.norm(values, ord = 2, axis = 0) # compute euclidean norm
    # values = values / norms # renormalize
    return values


def no_slip(x: np.ndarray) -> np.ndarray:
    # x hase shape (dimension, points)
    values = np.zeros((2, x.shape[1]))
    return values
