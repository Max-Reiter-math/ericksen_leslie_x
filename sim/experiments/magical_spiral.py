import os
from argparse import Namespace
from functools import partial
import numpy as np

from mpi4py import MPI
from dolfinx.io import XDMFFile
from sim.experiments.bcs_wo_fs import *
from sim.common.common_methods import set_attributes
from sim.common.common_fem_methods import angle_between
from sim.common.error_computation import *

"""
class for a standard benchmark setting for the ericksen-leslie model: 
    annihilation of two defects without an initial flow 
"""

class spiral:
    def __init__(self, args = Namespace()):
        self.name="magical spiral"
        # - model parameters namely v_el, const_A, nu, mu_1, mu_4, mu_5, mu_6, lam
        default_attributes = {"dim" : 2, "dh" : 10, "dt" : 0.001, "T" :  1.5, "t0": 0, "v_el": 1, "const_A":1.0, "nu":1.0,"mu_1":1.0, "mu_4": 1.0, "mu_5":1.0, "mu_6":1.0 , "lam":1.0, "algo2D": 6, "algo3D": 1}
        set_attributes(self, default_attributes, args)

        #SECTION - READ MESH
        mesh_loc = "input/meshes/spiral_"+str(self.dim)+"D_dh_"+str(self.dh)
        
        if os.path.isfile(mesh_loc+".xdmf"):
            # mesh exists in xdmf format
            with XDMFFile(MPI.COMM_WORLD, mesh_loc+".xdmf" , "r") as f:
                self.mesh = f.read_mesh()
                self.cell_tags = f.read_meshtags(self.mesh, name = "cell_tags")
                self.mesh.topology.create_connectivity(self.mesh.topology.dim - 1, self.mesh.topology.dim)
                self.meshtags = f.read_meshtags(self.mesh, name =  "facet_tags")

        elif os.path.isfile(mesh_loc+".msh"):
            # mesh exists in msh format
            from dolfinx.io import gmshio
            self.mesh, self.cell_tags, self.meshtags = gmshio.read_from_msh(mesh_loc+".msh", MPI.COMM_WORLD, 0, gdim=self.dim)

        else:
            raise FileNotFoundError("Could not find any mesh in msh or xdmf format under "+mesh_loc+"... To run this experiment the according mesh is needed as input.")
            
        inside, outside, up, down  =  self.meshtags.find(2) ,  self.meshtags.find(3) ,  self.meshtags.find(4),  self.meshtags.find(5)
        #!SECTION

        if self.dim == 2: self.boundary = boundary_2d
        elif self.dim == 3: self.boundary = boundary_3d
        d0 = partial(get_d0, dim = self.dim)
        v0 = partial(get_no_slip, dim = self.dim)
         # - initial conditions
        self.initial_conditions = {"v": v0, "p": (lambda x: np.full((x.shape[1],), 0.0)), "d": d0}
        # boundary conditions
        # dirichlet_bc_wo_fs("d", "topological", d0, entities = inside), \
        self.boundary_conditions = [dirichlet_bc_wo_fs("d", "topological", d0, entities = outside), \
                                    dirichlet_bc_wo_fs("d", "topological", d0, entities = inside), \
                                    dirichlet_bc_wo_fs("v", "topological", v0, entities= outside), \
                                    dirichlet_bc_wo_fs("v", "topological", v0, entities= inside)] #, \
                                    # component_dirichlet_bc_wo_fs("d", 2, "topological", partial(get_no_slip, dim = 1),  entities = up), \
                                    # component_dirichlet_bc_wo_fs("d", 2, "topological", partial(get_no_slip, dim = 1),  entities = down)]
    
    @property
    def info(self):
        return {"name":self.name}
    @property
    def has_exact_solution(self):
        if self.dim == 2: return True
        else: return False
    
    def exact_sol(self, x: np.ndarray) -> np.ndarray:
    # x hase shape (dimension, points)
        values = np.zeros(x.shape[1])
        r = np.sqrt(x[0]**2 + x[1]**2)    
        r0 = 1
        r1 = 2
        values = np.pi/2 * np.log(r/r0)/np.log(r1/r0)
        return values
    
    def compute_error(self, uh_, time, norm = "L2", degree_raise = 3):
        # arg time is not necessary for this class since we only consider the steady state limit of phi as exact solution
        # obtain FunctionSpace from approximate function
        # mesh   = uh_.function_space.mesh
        degree = uh_.function_space.ufl_element().degree()
        family = uh_.function_space.ufl_element().family()
        # degree = 1
        # family = "Lagrange"
        Q = FunctionSpace(self.mesh, (family, degree))
        Qr = FunctionSpace(self.mesh, (family, degree+degree_raise))
        D = VectorFunctionSpace(self.mesh, (family, degree), self.dim)
        # Compute and interpolate Angle
        # - compute radial lines first
        u_r, uh = Function(D), Function(D)
        u_r.interpolate(partial(unit_radials, dim = self.dim)) 

        # uh.x.array[:] = project(uh_,V, bcs= []):
        uh.interpolate(uh_)

        phi = Function(Q)
        phi.x.array[:] = angle_between(uh, u_r, self.dim)

        phir = Function(Qr)
        phir.interpolate(phi)
        uex = Function(Qr)
        uex.interpolate(self.exact_sol)


        err = np.sqrt(assemble_scalar(form(inner(phir-uex, phir-uex)*dx)))
        errinf = np.max(np.abs(uex.x.array[:] - phir.x.array[:] ))
        # err = errornorm(phi, self.exact_sol, norm = norm, degree_raise = degree_raise)
        
        return err

#SECTION - CUSTOM FUNCTIONS

def get_d0(x: np.ndarray, dim: int) -> np.ndarray:
    if dim not in [2,3]: raise ValueError("Dimension "+str(dim)+" not supported.")
    # x hase shape (dimension, points)

    values = np.zeros((dim, x.shape[1])) # values is going to be the output
    outside_dofs = bd_outside(x) # array of True and False giving the defect locations
    # rest = np.invert(outside_dofs)    

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

def boundary_3d(x: np.ndarray) -> np.ndarray:
    return np.logical_or.reduce((bd_disks(x), bd_inside(x), bd_outside(x)))

def boundary_2d(x: np.ndarray) -> np.ndarray:
    return np.logical_or(bd_inside(x), bd_outside(x))

def bd_disks(x):
    return np.logical_or(np.isclose(x[2],0),np.isclose(x[2],1))

def bd_outside(x):
    return np.isclose(x[0]**2 + x[1]**2, 4)

def bd_inside(x):
    return np.isclose(x[0]**2 + x[1]**2, 1, atol=.1)

#!SECTION
