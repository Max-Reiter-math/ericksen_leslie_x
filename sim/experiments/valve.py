import os
from argparse import Namespace
from functools import partial
import numpy as np
from basix.ufl import element
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary
from sim.common.meta_bcs import *
from sim.common.common_methods import set_attributes
from sim.common.common_fem_methods import angle_between
from sim.common.error_computation import *
from sim.common.mesh import circumcenters

"""
Microfluidic Valve

Usage examples:
    python -m sim.run -m linear_cg -e valve -vtx 1 -sid valve -dh 5 -fsr 0.0 -chi_perp -1.0 -beta 1.0 -gamma 1 -lam -1.2 -mu1 5 -mu5 6.8 -mu6 8 -dt 0.1
    python -m sim.run -m linear_cg -e valve -vtx 1 -sid valve -dh 5 -fsr 0.0 -gamma 1 -lam -1.2 -mu1 5 -mu5 6.8 -mu6 8 -mu4 0.1 -chi_vert -100.0 -dt 0.1 
"""

class valve:
    def __init__(self, comm, args = Namespace()):
        # NAME
        self.name="Valve"
        self.L = 3.0 # NOTE - make sure this coincides with the mesh
        self.height = 1.0
        
        self.vel_profile = args.vel_profile
        self.mag_angle = args.mag_angle


        # SECTION - MESH AND MESHTAGS
        mesh_loc = "input/meshes/valve_"+str(args.dh)
        
        if os.path.isfile(mesh_loc+".xdmf"):
            # mesh exists in xdmf format
            with XDMFFile(comm, mesh_loc+".xdmf" , "r") as f:
                self.mesh = f.read_mesh()
                self.mesh.topology.create_connectivity(self.mesh.topology.dim - 1, self.mesh.topology.dim)
                self.meshtags = f.read_meshtags(self.mesh, name =  "mesh_tags")

        else:
            raise FileNotFoundError("Could not find any mesh in msh or xdmf format under "+mesh_loc+"... To run this experiment the according mesh is needed as input.")
        
        self.dim = 2
        inlet, outlet1, outlet2, walls  =  self.meshtags.find(2) ,  self.meshtags.find(3), self.meshtags.find(4), self.meshtags.find(5)

        #DG0 int points
        if args.mod in ["linear_dg"]:
            self.dg0_cells, self.dg0_int_points = circumcenters(self.mesh)
        #!SECTION

        # INITIAL CONDITIONS
        self.initial_conditions = {"v": no_slip, "p": (lambda x: np.full((x.shape[1],), 0.0)), "d": self.homeotropic_anchoring, "H": self.electrical_field}

        # BOUNDARY CONDITIONS
        self.boundary_conditions = [
                                    meta_dirichletbc("d", "topological", self.homeotropic_anchoring, entities = walls, meshtag=5, marker = self.walls_marker),                                     # Dirichlet for director
                                    meta_dirichletbc("v", "topological", no_slip, entities= walls, meshtag=5),                                      # no slip bcs for velocity
                                    meta_dirichletbc("p", "topological", (lambda x: np.full((x.shape[1],), 0.0)),  entities= outlet1, meshtag=3),   # pressure
                                    meta_dirichletbc("p", "topological", (lambda x: np.full((x.shape[1],), 0.0)),  entities= outlet2, meshtag=4),   # pressure
                                    meta_dirichletbc("p", "topological", (lambda x: np.full((x.shape[1],), 1)),  entities= inlet, meshtag=2),       # pressure
                                    ]                                      
    @property
    def info(self):
        return {"name":self.name}
    @property
    def has_exact_solution(self):
        return False

    def homeotropic_anchoring(self, x: np.ndarray) -> np.ndarray:
        # x hase shape (dimension, points)
        values = np.zeros((2, x.shape[1])) # values is going to be the output

        inlet_channel_dofs = np.logical_and(x[0] < self.L, x[1] <= self.height/2, x[1] >= (-1)*self.height/2 )
        outlet_channel_dofs = x[0] >= self.L

        values[1][inlet_channel_dofs] += 1

        values[0][outlet_channel_dofs] += 1

        # renormalization
        norms = np.linalg.norm(values, ord = 2, axis = 0) # compute euclidean norm
        values = values / norms # renormalize
        return values
    
    def electrical_field(self, x: np.ndarray) -> np.ndarray:
        # x hase shape (dimension, points)
        values = np.zeros((2, x.shape[1])) # values is going to be the output

        # x hase shape (dimension, points)
        values = np.zeros((2, x.shape[1])) # values is going to be the output

        inlet_channel_dofs = x[0] < self.L
        upper_outlet_dofs = np.logical_and(x[0] >= self.L, x[1]>=self.height/2)
        lower_outlat_partial = np.logical_and(x[0] >= self.L, x[1]>=-1.0, x[1]<=-0.5)

        values[1][inlet_channel_dofs] += 1

        values[0][upper_outlet_dofs] += 1

        # renormalization
        # norms = np.linalg.norm(values, ord = 2, axis = 0) # compute euclidean norm
        # values = values / norms # renormalize
        return values
    
    def v0(self, x: np.ndarray) -> np.ndarray:
        # x hase shape (dimension, points)
        values = np.zeros((2, x.shape[1])) # values is going to be the output
        U = self.vel_profile
        values[0] = U * (1-(x[1]/self.height)**2)
        return values
    
    def inlet_marker(self, x: np.ndarray) -> np.ndarray:
        return np.isclose(x[0],0.0)

    def outlet_marker(self, x: np.ndarray) -> np.ndarray:
        return np.logical_or(np.isclose(x[1],0.0), self.height/2+self.L, np.isclose(x[1],0.0), (-1)*(self.height/2+self.L))
    
    def walls_marker(self,  x: np.ndarray) -> np.ndarray:
        inlet_walls = np.logical_and(np.logical_or(np.isclose(x[1], self.height/2), np.isclose(x[1], (-1)*(self.height/2))), x[0]<= self.L)

        outlet_walls = np.logical_or(np.logical_and(np.isclose(x[0],self.L), np.abs(x[1])>= self.height) ,  np.isclose(x[0], self.L + self.height))

        return np.logical_or(inlet_walls, outlet_walls)


def no_slip(x: np.ndarray) -> np.ndarray:
    # x hase shape (dimension, points)
    values = np.zeros((2, x.shape[1]))
    return values



