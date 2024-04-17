import numpy as np
from dolfinx.fem import ( FunctionSpace, Function, dirichletbc, locate_dofs_topological, locate_dofs_geometrical)
from dolfinx.mesh import meshtags, locate_entities
from ufl import (ds, inner, Measure)
"""
The following classes offer pre-initialized boundary conditions without the need to already specify the Function Space. This allows for a more flexible description of boundary conditions which can be then accessed by solvers which make use of a coupled or decoupled formulation.
"""
class dirichlet_bc_wo_fs():
    def __init__(self, quantity, locate_dofs, values, marker = None, entity_dim = None, entities = None):
        self.quantity = quantity   # e.g. "velocity" in order to differentiate the bcs for coupled quantities in PDEs
        self.type = "Dirichlet"
        self.find_dofs = locate_dofs
        self.marker = marker        # this is used if dofs are located geometrically
        self.entities = entities    # this is used if dofs are located topologically
        self.dim = entity_dim       # this is used if dofs are located topologically
        self.values = values
        
    def set_fs(self, FS, map = None):
        """
        example for mixed space should be written down here
        """
        if type(FS) == tuple:
            sub_FS = FS[0]
            collapsed_FS = FS[1]
        else:
            collapsed_FS = FS    
            sub_FS = FS
        u_D = Function(collapsed_FS)
        if self.find_dofs == "topological":
            boundary_dofs = locate_dofs_topological(FS, (collapsed_FS.mesh.topology.dim-1), self.entities)
        elif self.find_dofs == "geometrical":
            boundary_dofs = locate_dofs_geometrical(FS, self.marker)
        else: raise TypeError("Unknown method: {0:s}".format(self.find_dofs))
        u_D.interpolate(self.values) #, boundary_dofs)      # reduced dofs should be enough here, ask for this in fenics discourse
        if type(FS) == tuple:
            self.bc = dirichletbc(u_D, boundary_dofs, sub_FS)
        else:
            self.bc = dirichletbc(u_D, boundary_dofs)
        

class neumann_bc_wo_fs():
    def __init__(self, quantity, mesh, locator, values):
        self.quantity = quantity   # e.g. "velocity" in order to differentiate the bcs for coupled quantities in PDEs
        self.type = "Neumann"
        self.locator = locator
        self.values = values
        self.mesh = mesh
        
        
    def set_fs(self, FS, trial_func):    
        u_D = Function(FS)
        u_D.interpolate(self.values)
        #
        facet_indices, facet_markers = [], []
        fdim = self.mesh.topology.dim - 1
        facets = locate_entities(self.mesh, fdim, self.locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, 1))
        facet_indices = np.hstack(facet_indices).astype(np.int32)
        facet_markers = np.hstack(facet_markers).astype(np.int32)
        sorted_facets = np.argsort(facet_indices)
        facet_tag = meshtags(self.mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
        self.ds =  Measure("ds", domain=self.mesh, subdomain_data=facet_tag)
        self.v = trial_func
        self.bc = inner(u_D, self.v) * self.ds(1) #(self.marker)
        return self.bc

class robin_bc_wo_fs():
    def __init__(self, quantity, marker, values):
        self.quantity = quantity   # e.g. "velocity" in order to differentiate the bcs for coupled quantities in PDEs
        self.type = "Robin"
        self.marker = marker
        self.values = values
        
    def set_fs(self, test_func, trial_func):    
        self.u = test_func
        self.v = trial_func
        self.bc =  self.values[0] * inner(self.u-self.values[1], self.v)* ds(self.marker)
        return self.bc
        
class component_dirichlet_bc_wo_fs():
    def __init__(self, quantity: str, component: int, locate_dofs, values: callable, marker = None, entity_dim = None, entities = None):
        self.quantity = quantity   # e.g. "velocity" in order to differentiate the bcs for coupled quantities in PDEs
        self.type = "Dirichlet"
        self.component = component
        self.find_dofs = locate_dofs
        self.marker = marker        # this is used if dofs are located geometrically
        self.entities = entities    # this is used if dofs are located topologically
        self.dim = entity_dim       # this is used if dofs are located topologically
        self.values = values
        
    def set_fs(self, FS: FunctionSpace,  map = None):
        """
        example for mixed space should be written down here
        """
        if type(FS) == tuple:
            sub_FS = FS[0].sub(self.component)
            collapsed_FS , _ = FS[0].collapse()
        else:
            sub_FS = FS.sub(self.component)
            collapsed_FS = FS   
        sub_FS_collapsed , _ = sub_FS.collapse()
        u_D = Function(sub_FS_collapsed)
        if self.find_dofs == "topological":
            boundary_dofs = locate_dofs_topological((sub_FS, collapsed_FS), (collapsed_FS.mesh.topology.dim-1), self.entities)
        elif self.find_dofs == "geometrical":
            boundary_dofs = locate_dofs_geometrical((sub_FS, collapsed_FS), self.marker)
        else: raise TypeError("Unknown method: {0:s}".format(self.find_dofs))
        u_D.interpolate(self.values) #, boundary_dofs)      # reduced dofs should be enough here, ask for this in fenics discourse
        if type(FS) == tuple:
            self.bc = dirichletbc(u_D, boundary_dofs, sub_FS)
        else:
            self.bc = dirichletbc(u_D, boundary_dofs)