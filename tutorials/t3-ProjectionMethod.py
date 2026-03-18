# %% [markdown]
# # Tutorial 3: Projection Method
# In this example, we implement a projection method for the harmonic heat flow into the sphere. This yields a linear variational problem at each time step, followed by a nodal normalization step enforcing the unit-length constraint. This tutorial is based on the repo [harmonic_heat_flow](https://github.com/Max-Reiter-math/harmonic_heat_flow/blob/master/sim/models/linear_cg.py). For more mathematical details, consult **[Maximilian E. V. Reiter. (2025). Projection Methods in the Context of Nematic Crystal Flow](https://arxiv.org/abs/2502.08571)** for an application to liquid crystals.
# 
# Main contents:
# - Manual mesh generation,
# - manipulation of function coefficients / nodal values
# - error computation.
# 
# On the domain $\Omega \subset \mathbb{R}^{N}$, we denote the director field (local average of the molecules' orientation) by $ d: [0,T] \times \overline{\Omega} \to \mathbb{R}^{N}$. The harmonic heat flow into the sphere is governed by the equations:
# 
# $$
# \begin{aligned}
# \partial_t d - \gamma (I - d \otimes d)  \Delta d   & = 0 , \\
# \vert d \vert^2 & = 1 .
# \end{aligned}
# $$
# 
# We equip the system with constant-in-time Dirichlet boundary conditions,
# $$
# d(t) = d_0 \text{ on } \partial \Omega \text{ for all }t\in(0,T) \, . 
# $$

# %% [markdown]
# ## Step 1: Mesh Generation
# We create a disk with a centered hole as domain, i.e. $\Omega = \{x \in \mathbb{R}^2: 1<\vert x \vert < 2\}$. While FEniCSx offers easy constructors for simple rectangular or cube-like domains, more complicated meshes have to be built using unstructured methods (see e.g. GMSH) or specified manually. The latter approach is shown here exemplary for the above domain.

# %%
import numpy as np
from mpi4py import MPI
import ufl
import basix
from dolfinx.mesh import create_mesh

def donut_mesh(comm: MPI.Comm, n: int):
    """
    Creates a mesh for a disk with radius r2 = 2, with a hole with radius r1 = 1.
    """
    n_radius = np.max([2,n]) # at least vertices on the inner and on the outer boundary are needed for the mesh to be well-defined
    n_angle  = np.max([8,4*n]) # at least two points per quarter are needed
    
    # arranging points along partition of the radius as well as the angle
    angles      = np.linspace(0.0, 2*np.pi, n_angle, endpoint = False)
    radii       = np.linspace(1.0, 2.0,     n_radius)

    pts_x   = np.outer( radii, np.cos(angles)).flatten() 
    pts_y   = np.outer( radii, np.sin(angles)).flatten() 
    pts     = np.array([ pts_x, pts_y ])
    """
    The resulting points are indexed the following way:
    index from 0 to n_angle-1 : points on the inner boundary starting at (1,0)
    ...
    index from (n_radius-1)*n_angle to n_radius*n_angle-1 : points on the outer boundary starting at (2,0)

    The cells now consist of triangles defined by the indices of their three vertices.
    To create them we iterate/vectorize through the angle and radius partition and create the left and right pointing subsequently.

    Left Pointing     Right Pointing
    |<----|             |>----|
    | <<xx|             |x>>  |
    |  <<x|             |xx>> |
    |   <x|             |xxx> |
    |-----|             |-----|

    Both the left and right pointing triangles can be defined by the four points containing them.
    """
    top_left        = np.arange(0, (n_radius-1)*n_angle, 1) + n_angle
    top_right       = np.array([np.roll(np.arange(0, n_angle, 1), -1) + i*n_angle for i in range(1, n_radius, 1)]).flatten()
    bottom_left     = np.arange(0, (n_radius-1)*n_angle, 1)
    bottom_right    = np.array([np.roll(np.arange(0, n_angle, 1), -1) + i*n_angle for i in range(0, n_radius-1, 1)]).flatten()
    # print(bottom_left)
    # print(bottom_right)
    # print(top_left)
    # print(top_right)

    cells_r = np.array([top_left, bottom_left, bottom_right])
    cells_l = np.array([top_left, top_right, bottom_right])
    cells = np.hstack([cells_r , cells_l])
    # print(cells.T)

    ufl_mesh    = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))) # shape yields the geometric dimension
    domain      = create_mesh(comm, cells.T, pts.T, ufl_mesh)

    return domain

domain = donut_mesh(MPI.COMM_WORLD, 15)
dim    = 2

# %% [markdown]
# ### Optional: Visualization using Pyvista
# Note that the resulting mesh is not exactly non-obtuse, i.e. the angles of the triangles may exceed 90 degrees. However, the finer the mesh is, the less this may affect the method as for weakly acute meshes small deviations are allowed.

# %%

import pyvista
import dolfinx.plot as plot

from dolfinx import plot

domain.topology.create_connectivity(dim, dim)
topology, cell_types, geometry = plot.vtk_mesh(domain, dim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, style="wireframe", color="black", line_width=2)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("t3-mesh.png")

# %% [markdown]
# ## Optional: Check if Mesh is Weakly Acute
# Moreover, we can explicitly check the prerequisite for the energy-decreasing projection method, that is a weakly acute mesh. A weakly acute mesh is defined by the stiffness matrix having only non-positive non-diagonal entries.

# %%
from ufl import inner, dx, grad, TestFunction, TrialFunction
from dolfinx.fem import form
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.fem import Function, functionspace, ElementMetaData

FS  = functionspace(domain, ElementMetaData("Lagrange", 1 , shape=(dim,)))
u   = TrialFunction(FS)
phi = TestFunction(FS)

stiffnes_matrix = form(inner(grad(u), grad(phi))*dx)
M = assemble_matrix(stiffnes_matrix)
M.assemble()
M.convert("dense")
S = np.asarray(M.getDenseArray()).round(decimals = 2)
print("Stiffness Matrix:", S)

off_diagonal_elements = S[np.invert(np.eye(S.shape[0], dtype=bool))]

# Assert off-diagonal elements are zero or negative
if not np.all(off_diagonal_elements <= 0):
    raise AssertionError("Off-diagonal elements must be zero or negative")

print("Assertion passed: matrix has non-positive entries except for the diagonal.")

# %% [markdown]
# ## Step 2: Initialization of function spaces and functions
# For the discrete director field globally continuous, affine-linear elements (CG1) are chosen. Its consistent approximation of the Laplacian using an auxiliary variable is also achieved using globally continuous, affine-linear elements (CG1). This can be implemented in FEniCSx using a mixed method. Although one could assemble the linearized system on a mixed product space, in practice it is often more convenient and robust to work with separate blocks for $d$ and $q$, especially when configuring block solvers and boundary conditions.

# %%
from ufl import TestFunction, TrialFunction
from dolfinx.fem import Function, functionspace, ElementMetaData

D = functionspace(domain, ElementMetaData("Lagrange", 1 , shape=(dim,))) # FE space for the director field
Y = functionspace(domain, ElementMetaData("Lagrange", 1 , shape=(dim,))) # FE space for the discrete Laplacian

d1, q1                       = TrialFunction(D), TrialFunction(Y)   # unknowns
d0, q0                       = Function(D), Function(Y)             # knowns
d,  q                        = Function(D), Function(Y)             # result vector
c, b                         = TestFunction(D), TestFunction(Y)

# %% [markdown]
# ## Step 3: Variational Formulation of the Projection Method
# For the local conservation properties of our scheme, we once again need the mass-lumped inner product, defined on $[CG1]^N$ by
# $$
# ( f_h, g_h)_h 
# =
# \int_{\Omega} \mathcal{I}_h^1 ( f_h \cdot  g_h) \mathrm{d} x
# =
# \sum_{{z} \in \mathcal{N}}  f_h ({z}) \cdot  g_h ({z}) \int_{\Omega} \Phi_{{z}} \mathrm{d} x
# \, .
# $$

# %%
from ufl import Measure
dxL = Measure("dx", domain = domain, metadata = {"quadrature_rule": "vertex", "quadrature_degree": 0})  # mass lumping

# %% [markdown]
# In the projection method, the constraint $\vert d \vert^2 = 1$ is not enforced directly in the linear solve, but restored afterwards by normalization. Let $k>0$ be the time-step size. For $j=1,2,...$, we **first** solve a fully linear system given by
# $$
# \begin{aligned}
# \frac{1}{k} (d^j -d^{j-1}, c)_h + \gamma  \left((I - d^{j-1} \otimes d^{j-1}) q^{j}  ,c\right)_h & = 0 , \\
# (q^{j}, b)_h - (\nabla d^{j}, \nabla b)_2 &= 0 \, ,
# \end{aligned}
# $$
# for all $(b,c)\in [CG1]^N\times [CG1]^N$. Hereby, the variational derivative $q$ has to be interpreted as $-\Delta d$.
# 
# In the **second** step, we normalize the solution at every node $z \in \mathcal{N}$ by
# $$
# d^j (z) \leftarrow \frac{d^j (z)}{\vert d^j (z) \vert} \, .
# $$
# 
# Since we do not use a monolithic approach for the linear problem, we have to define each block in the linear system individually. Schematically, the corresponding block operator has the form
# 
# $$
# \begin{pmatrix}
# \frac{1}{k} (\cdot,c)_h
# &
# +\gamma \left((I  - d^{j-1} \otimes d^{j-1})   \cdot ,c \right)_h
# \\
# - (\nabla \cdot, \nabla b)_2
# &
# +(\cdot, b)_h
# \end{pmatrix}
# \cdot
# \begin{pmatrix}
# d^{j}\\
# q^{j}
# \end{pmatrix}
# =
# \begin{pmatrix}
# \frac{1}{k} (d^{j-1},c)_h\\
# 0
# \end{pmatrix}
# \, .
# $$

# %%
from petsc4py import PETSc
from ufl import inner, dx, grad
from dolfinx.fem import form, Constant

k       = 0.01     # time-step size
gamma   = 1.0       # damping parameter

# heat flow
a11  = inner(d1, c)*dxL                             # discrete time derivative
L1   = inner(d0, c)*dxL                             # discrete time derivative
a12  = k*gamma *inner(q1, c)*dxL                    # damping term
a12 -= k*gamma *inner(q1, d0)*inner(d0, c)*dxL      # damping term
# equation for auxiliary variable
a21  = -inner(grad(d1),grad(b))*dx
a22  = inner(q1, b)*dxL
L2   = inner(Constant(domain, (PETSc.ScalarType(0), PETSc.ScalarType(0))), b)*dx

a = form([
        [a11, a12],
        [a21, a22]
    ])

L = form([
        L1 ,
        L2 ,
    ]) 

# %% [markdown]
# We also prepare the nodal normalization by defining the according method. Note that for CG1 functions the coefficients correspond to the vertex and nodal values. All these notions coincide.

# %%
def nodal_normalization(d, dim):
    coeffs = np.reshape( d.x.array[:] , (-1, dim))                      # has shape (#nodes, dim)
    norms = np.linalg.norm(coeffs, axis=1, keepdims=True).flatten()     # compute magnitude of nodal values
    coeffs[:] = coeffs[:] / norms[:, np.newaxis]
    # Overwrite coefficients
    d.x.array[:] = np.reshape(coeffs, (-1,))

# %% [markdown]
# ## Step 4: Initial Values
# We only need the initial director field $d_0$, since the scheme does not require a previous-time-step value for the auxiliary variable $q$. The initial director field is given by the radial direction, i.e.
# $$
# d_0 (x)  = x / \vert x \vert \text{ in }\Omega\, .
# $$
# Note that we already include the definition of the boundary conditions here, given by
# $$
# d_0 (x)  = (x_2, -x_1)^T / \vert x \vert \text{ for all } \{x : \vert x \vert = 2\} \, ,
# \qquad 
# d_0 (x)  = x / \vert x \vert \text{ for all } \{x : \vert x \vert = 1 \} \, .
# $$

# %%
def bd_outside(x):
    return np.isclose(x[0]**2 + x[1]**2, 4)

def bd_inside(x):
    return np.isclose(x[0]**2 + x[1]**2, 1)

def get_d0(x: np.ndarray) -> np.ndarray:
    # x hase shape (dimension, points)
    values = np.zeros((dim, x.shape[1])) # values is going to be the output
    outside_dofs = bd_outside(x) # array of True and False giving the defect locations  

    # Setting 
    # - normal to the boundary with some tilt described by eta
    values[0]=x[0] 
    values[1]=x[1] 

    # Setting outside BC    
    # - tangential to sphere 
    values[0][outside_dofs]=x[1][outside_dofs]
    values[1][outside_dofs]=-x[0][outside_dofs] 

    # renormalization
    norms = np.linalg.norm(values, ord = 2, axis = 0) # compute euclidean norm
    values = values / norms # renormalize
    return values

d0.interpolate(get_d0) # interpolation of d^0

# %% [markdown]
# ## Step 5: Boundary Conditions
# The boundary condition is already defined by the _callable_ **get_d0**. We need to locate the according degrees of freedom (DOFs). This can be done using a geometric description of the boundary.

# %%
from dolfinx.fem import locate_dofs_geometrical, dirichletbc

dofs_inside     = locate_dofs_geometrical(D, bd_inside)
dofs_outside    = locate_dofs_geometrical(D, bd_outside)

d_initial = Function(D)
d_initial.interpolate(get_d0)

bcs = [ dirichletbc(d_initial, dofs_inside), dirichletbc(d_initial, dofs_outside) ]

# %% [markdown]
# ## Step 6: Problem and Solver Setup
# We set up the linear problem and its solver by assembling the matrix and right-hand side vector and solving the resulting linear problem.

# %%
from petsc4py import PETSc
import dolfinx.la as la
from dolfinx.fem import Function, bcs_by_block, extract_function_spaces, bcs_by_block
from dolfinx.fem.petsc import assemble_matrix_nest, assemble_vector_nest, apply_lifting_nest, set_bc_nest, set_bc_nest

def assemble_and_solve(d, q, solver):
    # Assemble nested matrix operators
    A = assemble_matrix_nest(a, bcs=bcs)
    A.assemble()

    # Assemble right-hand side vector
    b = assemble_vector_nest(L)

    apply_lifting_nest(b, a, bcs=bcs)                       # modifies the assembled right-hand side to account for the effect of Dirichlet conditions that have been imposed on the matrix blocks, so the linear system remains algebraically consistent
    bcs0 = bcs_by_block(extract_function_spaces(L), bcs)    # reorganizes the boundary conditions by block/function space,
    set_bc_nest(b, bcs0)                                    # inserts the prescribed boundary values into the corresponding entries of the nested right-hand side vector.

    x = PETSc.Vec().createNest([la.create_petsc_vector_wrap(d.x), la.create_petsc_vector_wrap(q.x)])
    
    solver.setOperators(A)
    solver.solve(b, x)

# Setting up sequential Krylov solver
ksp = PETSc.KSP().create(MPI.COMM_WORLD)
ksp.setType("gmres")
ksp.setTolerances(rtol=1e-9)
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType(PETSc.Mat.SolverType.MUMPS)

# %% [markdown]
# ## Step 7: Preprocessing for the Temporal Evolution
# Setup Output Pipeline for the director and velocity field and save for $t = 0$. As output format, we use the vtx format this time instead of the xdmf format. 

# %%
from dolfinx.fem import form, assemble_scalar
from dolfinx.io import VTXWriter

vtx_d = VTXWriter(MPI.COMM_WORLD, "t3-d.bp", d0, engine="BP4")

vtx_d.write(0.0)

print("Time: 0.0.")

def energy(d):
    return assemble_scalar(form(   0.5 *  inner(grad(d), grad(d))*dx   ))

e_ela = energy(d0)
print(f"Elastic Energy: {e_ela}.")

# %% [markdown]
# For this setting, the exact stationary solution is known and given in terms of its angle between the director field and the radial direction,
# $$
# \psi (x) = \frac{\pi}{2} \frac{\log (\vert x\vert)}{\log (2)} \, .
# $$
# Consequently, we define the exact angle of the solution and prepare the error computation.

# %%
from ufl import SpatialCoordinate, acos, sqrt, conditional
from dolfinx.fem import Expression

CG          = functionspace(domain, ElementMetaData("Lagrange", 3)) # NOTE that we increase the polynomial order to compute the error more exact
psi_exact   = Function(CG)
psi         = Function(CG)

def exact_solution(x: np.ndarray) -> np.ndarray:
    # x has shape (dimension, points)
    r       = np.sqrt(x[0]**2 + x[1]**2)   
    angles  = np.pi/2 * np.log(r)/np.log(2)
    return angles

psi_exact.interpolate(exact_solution)
    
coord = SpatialCoordinate(domain)

def compute_error(d):
    """
    The angle $\theta$ between two vectors $v$ and $w$ can be computed as
    $\cos \theta = (v,w)/(|v||w|)$
    """
    cos_theta       = (coord[0]*d[0] + coord[1]*d[1])/(sqrt(coord[0]**2 + coord[1]**2)*sqrt(d[0]**2 + d[1]**2))
    cos_theta_safe  = conditional(cos_theta < 1.0, cos_theta , 1.0)
    theta_safe      = acos(cos_theta_safe)    
    expr = Expression(
        theta_safe,
        CG.element.interpolation_points()
    )
    psi.interpolate(expr)

    err   = assemble_scalar(form(inner(psi-psi_exact,psi-psi_exact)*dx))
    return err

err0 = compute_error(d0)
print(f"Initial Squared L2 Error: {err0}.")

# %% [markdown]
# For more on error computation, consult this [tutorial](https://jsdokken.com/dolfinx-tutorial/chapter4/convergence.html).

# %% [markdown]
# ## Step 8: Time evolution
# In every time step, we first solve the equation system and then update the resulting director field.

# %%
t = 0.0
T = 1.0

while t < T:
    t += k 

    print(f"Time Step: {t}.")
        
    assemble_and_solve(d, q, ksp)

    # properties before nodal normalization
    e_ela = energy(d)
    unit_max = np.max(np.linalg.norm(np.reshape( d.x.array[:] , (-1, dim)), axis=1))    
    unit_min = np.min(np.linalg.norm(np.reshape( d.x.array[:] , (-1, dim)), axis=1))
    print(f"Before Normalization - Elastic Energy: {e_ela}; max norm of director field (min - max): ( {unit_min} - {unit_max} )")

    # Normalization
    nodal_normalization(d, dim)
    
    # properties before nodal normalization
    e_ela = energy(d)
    unit_max = np.max(np.linalg.norm(np.reshape( d.x.array[:] , (-1, dim)), axis=1))    
    unit_min = np.min(np.linalg.norm(np.reshape( d.x.array[:] , (-1, dim)), axis=1))
    print(f"After Normalization - Elastic Energy: {e_ela}; max norm of director field (min - max): ( {unit_min} - {unit_max} )")

    # Error computation
    err = compute_error(d)
    print(f"Squared L2 Error: {err}.\n")

    # update and save director field
    d0.x.array[:] = d.x.array[:] # NOTE that we don't update q0 as it does not show up in the variational formulation
    vtx_d.write(t)

vtx_d.close()

# %% [markdown]
# ## Optional: Visualize the Results
# Using Pyvista, we compare the exact stationary solution and the simulated result at $t=T$.

# %%
cells, types, x = plot.vtk_mesh(CG)    
grid = pyvista.UnstructuredGrid(cells, types, x)  

grid.point_data["u"] = psi_exact.x.array
grid.point_data["uh"] = psi.x.array

grid.set_active_scalars("u")
subplotter = pyvista.Plotter(shape=(1, 2))
subplotter.subplot(0, 0)
subplotter.add_text("Exact Solution", font_size=14, color="black", position="upper_edge")
sargs = dict(height=0.1, width=0.8, vertical=False, position_x=0.1, position_y=0, color="black")
subplotter.add_mesh(grid, show_edges=False, scalar_bar_args=sargs, line_width=0)
subplotter.view_xy()

grid.set_active_scalars("uh")
subplotter.subplot(0, 1)
subplotter.add_text("Computed solution", font_size=14, color="black", position="upper_edge")
sargs = dict(height=0.1, width=0.8, vertical=False, position_x=0.1, position_y=0, color="black")
subplotter.add_mesh(grid, show_edges=False, scalar_bar_args=sargs, line_width=0)
subplotter.view_xy()

if pyvista.OFF_SCREEN:
    subplotter.screenshot(
        "t3-comparison.png",
        transparent_background="transparent",
        window_size=[1500, 400],
    )
else:
    subplotter.show()


