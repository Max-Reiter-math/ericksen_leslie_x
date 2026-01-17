import numpy as np
from dolfinx.fem import ( FunctionSpace, Function, dirichletbc, locate_dofs_topological, locate_dofs_geometrical)
from dolfinx.mesh import meshtags, locate_entities
from ufl import (ds, inner, Measure)
"""
The following classes offer pre-initialized boundary conditions without the need to already specify the Function Space. This allows for a more flexible description of boundary conditions which can be then accessed by solvers which make use of a coupled or decoupled formulation.
"""

class meta_dirichletbc():
    def __init__(self, quantity: str, locate_dofs: str, values: callable, marker = None, entity_dim = None, entities = None, meshtag = 0):
        self.type = "Dirichlet"
        self.quantity = quantity            # e.g. "velocity" in order to differentiate the bcs for coupled quantities in PDEs
        self.find_dofs = locate_dofs        # either "topological" or "geometrical" for the two options reflected in fenicsx
        self.marker = marker                # if dofs are located geometrically this is the implicit function describing the area
        self.entities = entities            # this is used if dofs are located topologically
        self.dim = entity_dim               # this is used if dofs are located topologically
        self.values = values                # this is the callable to prescribe the conditions
        self.meshtag = meshtag              # if no meshtag is given the whole domain of the boundary will be assumed

    def __str__ (self):
        return str({"type" : self.type, "quantity" : self.quantity, "find_dofs" : self.find_dofs, "marker" : self.marker, "entitites" : self.entities, "dim" : self.dim, "values": self.values, "meshtag" : self.meshtag})
        
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
        self.u_D = Function(collapsed_FS)

        if self.find_dofs == "topological":
            self.boundary_dofs = locate_dofs_topological(V = FS, entity_dim = (collapsed_FS.mesh.topology.dim-1), entities = self.entities)
        elif self.find_dofs == "geometrical":
            self.boundary_dofs = locate_dofs_geometrical(FS, self.marker)
        else: 
            raise TypeError("Unknown method: {0:s}".format(self.find_dofs))
        
        self.u_D.interpolate(self.values) #, boundary_dofs)      # reduced dofs should be enough here, ask for this in fenics discourse
        if type(FS) == tuple:
            self.bc = dirichletbc(self.u_D, self.boundary_dofs, sub_FS)
        else:
            self.bc = dirichletbc(self.u_D, self.boundary_dofs)
        
class meta_componentdirichletbc(meta_dirichletbc):
    def __init__(self, quantity: str, locate_dofs: str, values: callable, marker = None, entity_dim = None, entities = None, meshtag = 0, component = 0):
        super().__init__(self, quantity, locate_dofs, values, marker = marker, entity_dim = entity_dim, entities = entities, meshtag = meshtag) 
        self.type = "Dirichlet"
        self.component = component
        
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
        self.u_D = Function(sub_FS_collapsed)
        if self.find_dofs == "topological":
            boundary_dofs = locate_dofs_topological((sub_FS, collapsed_FS), (collapsed_FS.mesh.topology.dim-1), self.entities)
        elif self.find_dofs == "geometrical":
            boundary_dofs = locate_dofs_geometrical((sub_FS, collapsed_FS), self.marker)
        else: raise TypeError("Unknown method: {0:s}".format(self.find_dofs))
        self.u_D.interpolate(self.values) #, boundary_dofs)      # reduced dofs should be enough here, ask for this in fenics discourse
        if type(FS) == tuple:
            self.bc = dirichletbc(self.u_D, boundary_dofs, sub_FS)
        else:
            self.bc = dirichletbc(self.u_D, boundary_dofs)

class meta_neumannbc():
    #TODO - Needs rework
    def __init__(self, quantity: str, values: callable, marker = None, meshtags = None, meshtag_id = 0):
        self.type = "Neumann"
        self.quantity = quantity            # e.g. "velocity" in order to differentiate the bcs for coupled quantities in PDEs
        self.values = values                # this is the callable to prescribe the conditions
        # either geometric marking of the boundary
        self.marker = marker                # if dofs are located geometrically this is the implicit function describing the area
        # or by an explicit meshtag
        self.meshtags = meshtags
        self.meshtag_id = meshtag_id           # if no meshtag is given the whole domain of the boundary will be assumed
        
        
    def set_fs(self, FS, trial_func): 
        self.u_D = Function(FS)
        self.u_D.interpolate(self.values)
        #
        if meshtags == None:
            facet_indices, facet_markers = [], []
            fdim = FS.mesh.topology.dim - 1
            facets = locate_entities(FS.mesh, fdim, self.locator)
            facet_indices.append(facets)
            facet_markers.append(np.full_like(facets, 1))
            facet_indices = np.hstack(facet_indices).astype(np.int32)
            facet_markers = np.hstack(facet_markers).astype(np.int32)
            sorted_facets = np.argsort(facet_indices)
            facet_tag = meshtags(self.mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
            self.ds =  Measure("ds", domain=self.mesh, subdomain_data=facet_tag)
        else:
            self.ds = Measure("ds", domain=self.mesh, subdomain_data=self.meshtags)
        self.v = trial_func
        self.bc = inner(self.u_D, self.v) * self.ds(self.meshtag_id)

class meta_robinbc():    
    #TODO - Needs rework
    def __init__(self, quantity, marker, values):
        self.quantity = quantity   # e.g. "velocity" in order to differentiate the bcs for coupled quantities in PDEs
        self.type = "Robin"
        self.marker = marker
        self.values = values
        
    def set_fs(self, test_func, trial_func):    
        # TODO - rework
        self.u = test_func
        self.v = trial_func
        self.bc =  self.values[0] * inner(self.u-self.values[1], self.v)* ds(self.marker)
        
