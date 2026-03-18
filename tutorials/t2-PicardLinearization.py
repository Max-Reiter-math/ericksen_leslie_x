# %% [markdown]
# # Tutorial 2: Picard-Type Linearization
# In this example, we implement a Picard-type linearization of the non-linear method for the harmonic heat flow into the sphere presented in tutorial 1, resulting in a fixed-point scheme. This tutorial is based on the repo [harmonic_heat_flow](https://github.com/Max-Reiter-math/harmonic_heat_flow/blob/master/sim/models/fp_coupled.py).
# 
# Main contents:
# - Implementation of a (nested) linear problem,
# - introduction of a Picard-type fixed-point linearization,
# - configuration of Dirichlet boundary conditions.
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
# We equip the system with constant-in-time Dirichlet boundary conditions, equal to the trace of the initial director field, i.e.
# $$
# d(t) = d_0 \text{ on } \partial \Omega \text{ for all }t\in(0,T) \, . 
# $$

# %% [markdown]
# ## Step 1: Mesh Generation
# We create the domain $\Omega = (-1,1)^2$.

# %%
import numpy as np
from dolfinx.mesh import create_rectangle, CellType
from dolfinx.cpp.mesh import DiagonalType
from mpi4py import MPI

dim = 2
n   = 5

domain = create_rectangle(MPI.COMM_WORLD, [np.array([-1.0, -1.0]), np.array([1.0, 1.0])],  [n, n], cell_type = CellType.triangle, diagonal=DiagonalType.left_right)

# %% [markdown]
# ## Step 2: Initialization of Function Spaces and Functions
# For the discrete director field globally continuous, affine-linear elements (CG1) are chosen. Its consistent approximation of the Laplacian using an auxiliary variable is also achieved using globally continuous, affine-linear elements (CG1). This can be implemented in FEniCSx using a mixed method. Although one could assemble the linearized system on a mixed product space, in practice it is often more convenient and robust to work with separate blocks for $d$ and $q$, especially when configuring block solvers and boundary conditions.

# %%
from ufl import TestFunction, TrialFunction
from dolfinx.fem import Function, functionspace, ElementMetaData

D = functionspace(domain, ElementMetaData("Lagrange", 1 , shape=(dim,))) # FE space for the director field
Y = functionspace(domain, ElementMetaData("Lagrange", 1 , shape=(dim,))) # FE space for the discrete Laplacian

d1, q1                       = TrialFunction(D), TrialFunction(Y)   # unknowns
d0, q0                       = Function(D), Function(Y)             # knowns
d,  q                        = Function(D), Function(Y)             # current iterate
dl, ql                       = Function(D), Function(Y)             # previous iterate in fixed point scheme
c, b                         = TestFunction(D), TestFunction(Y)

# %% [markdown]
# We further introduce the following midpoint discretization $d^{j-1/2,l-1} = \frac{1}{2} d^{j,l-1}+ \frac{1}{2} d^{j-1}$.

# %%
dl_ = 0.5*dl + 0.5*d0

# %% [markdown]
# ## Step 3: Variational Formulation
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
# Let $k>0$ be the time-step size. The non-linear formulation reads for $j=1,2,...$ as
# $$
# \begin{aligned}
# \frac{1}{k} (d^j -d^{j-1}, c)_h + \gamma  \left((I \vert d^{j-1/2}  \vert^2 - d^{j-1/2} \otimes d^{j-1/2}) q^{j}  ,c\right)_h & = 0 , \\
# (q^{j}, b)_h - (\nabla d^{j-1/2}, \nabla b)_2 &= 0 \, ,
# \end{aligned}
# $$
# for all $(b,c)\in [CG1]^N\times [CG1]^N$. Hereby, the variational derivative $q$ has to be interpreted as $-\Delta d$.
# For its Picard-Type linearization, we discretize the projection matrix $\mathrm{I} - d \otimes d$ with the previous Picard iterate. Let $j=1,2,...$ be fixed. For $l=1,2,...$, we solve the equation system
# $$
# \begin{aligned}
# \frac{1}{k} (d^{j,l} -d^{j-1},c)_h + \gamma \left((I \vert d^{j-1/2,l-1}  \vert^2 - d^{j-1/2,l-1} \otimes d^{j-1/2,l-1})   q^{j,l} ,c \right)_h & = 0 , \\
# (q^{j,l}, b)_h - (\nabla d^{j-1/2,l}, \nabla b)_2 &= 0 \, ,
# \end{aligned}
# $$
# for all $(b,c)\in [CG1]^N\times [CG1]^N$ with $d^{j,0}=d^{j-1}$.
# 
# Since we do not use a monolithic approach for the linear problem, we have to define each block in the linear system individually. Schematically, the corresponding block operator has the form
# 
# $$
# \begin{pmatrix}
# (\cdot,c)_h
# &
# + k \gamma \left((I \vert d^{j-1/2,l-1}  \vert^2 - d^{j-1/2,l-1} \otimes d^{j-1/2,l-1})   \cdot ,c \right)_h
# \\
# - \frac{1}{2} (\nabla \cdot, \nabla b)_2
# &
# +(\cdot, b)_h
# \end{pmatrix}
# \cdot
# \begin{pmatrix}
# d^{j,l}\\
# q^{j,l}
# \end{pmatrix}
# =
# \begin{pmatrix}
# (d^{j-1},c)_h\\
# \frac{1}{2} (\nabla d^{j-1}, \nabla b)_2
# \end{pmatrix}
# \, .
# $$
# 
# Note that the director equation was multiplied by the time step size for improved computational behaviour.

# %%
from ufl import inner, dx, grad
from dolfinx.fem import form

k       = 0.01     # time-step size
gamma   = 1.0       # damping parameter

# heat flow
a11  = inner(d1, c)*dxL                             # discrete time derivative
L1   = inner(d0, c)*dxL                             # discrete time derivative
a12  = k*gamma *inner(q1, c)*inner(dl_, dl_)*dxL    # damping term
a12 -= k*gamma *inner(q1, dl_)*inner(dl_, c)*dxL    # damping term
# equation for auxiliary variable
a21  = -0.5*inner(grad(d1),grad(b))*dx
a22  = inner(q1, b)*dxL
L2   = 0.5*inner(grad(d0),grad(b))*dx

a = form([
        [a11, a12],
        [a21, a22]
    ])

L = form([
        L1 ,
        L2 ,
    ]) 

# %% [markdown]
# ## Step 4: Initial Values
# We only need the initial director field $d_0$, since the scheme does not require a previous-time-step value for the auxiliary variable $q$.

# %%
def get_d0(x: np.ndarray)-> np.ndarray:
    # x has shape (dimension, points)
    values = np.zeros((dim, x.shape[1])) # values is going to be the output
    
    # Setting defects
    values[0]= np.sin( 2.0*np.pi*(np.cos(x[0])-np.sin(x[1]) ) )
    values[1]= np.cos( 2.0*np.pi*(np.cos(x[0])-np.sin(x[1]) ) )

    # renormalization
    norms = np.linalg.norm(values, ord = 2, axis = 0) # compute euclidean norm
    values = values / norms # renormalize
    return values

d0.interpolate(get_d0) # interpolation of d^0
dl.interpolate(get_d0) # interpolation of d^{j,0} = d^{j-1} = d^0

# %% [markdown]
# ## Step 5: Boundary Conditions
# To do so, we need to locate the according degrees of freedom (DOFs). This can be done using a geometric description of the boundary. Since the Dirichlet data are time-independent and equal to the initial trace, we interpolate the initial director field and use it as the boundary value throughout the simulation.

# %%
from dolfinx.fem import locate_dofs_geometrical, dirichletbc

def boundary_2d(x: np.ndarray) -> np.ndarray:
    return np.logical_or.reduce((np.isclose(x[0], -1.0), np.isclose(x[0], 1.0), np.isclose(x[1], -1.0), np.isclose(x[1], 1.0)))

dofs_D = locate_dofs_geometrical(D, boundary_2d)

d_initial = Function(D)
d_initial.interpolate(get_d0)

bcs = [ dirichletbc(d_initial, dofs_D) ]

# %% [markdown]
# ## Step 6: Problem and Solver Setup
# We set up the linear problem and its solver by assembling the matrix and right-hand side vector and solving the resulting linear problem.

# %%
from petsc4py import PETSc
import dolfinx.la as la
from dolfinx.fem import bcs_by_block, extract_function_spaces, bcs_by_block
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

# Setting up a Krylov solver for the nested linear system.
ksp = PETSc.KSP().create(MPI.COMM_WORLD)
ksp.setType("gmres")
ksp.setTolerances(rtol=1e-9)
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType(PETSc.Mat.SolverType.MUMPS)

# %% [markdown]
# Setup Output Pipeline for the director and velocity field and save for $t = 0$. Due to the mixed space, we first interpolate the functions into the collapsed space before writing to file. As output format, we use the vtx format this time instead of the xdmf format. 

# %%
from dolfinx.fem import form, assemble_scalar
from dolfinx.io import VTXWriter

vtx_d = VTXWriter(MPI.COMM_WORLD, "t2-d.bp", d0, engine="BP4")

vtx_d.write(0.0)

print("Time: 0.0.")

def energy(d):
    return assemble_scalar(form(   0.5 *  inner(grad(d), grad(d))*dx   ))

e_ela = energy(d0)
print(f"Elastic Energy: {e_ela}.")

# %% [markdown]
# Since we consider globally continuous, piecewise linear functions (CG1), their coefficients, the nodal and the vertex evaluations of the functions coincide. Accordingly, we can compute the maximal deviation from the unit-sphere constraint by a nodal evaluation:

# %%
# compute agreement with unit-sphere constraint
unit_max = np.max(np.linalg.norm(np.reshape( d0.x.array[:] , (-1, dim)), axis=1))    
unit_min = np.min(np.linalg.norm(np.reshape( d0.x.array[:] , (-1, dim)), axis=1))
print(f"max norm of director field (min - max): ( {unit_min} - {unit_max} )")

# %% [markdown]
# ## Step 7: Time evolution
# In every time step, we solve the fixed-point scheme until the relative or absolute stop criterion in terms of the energy norm is reached, i.e.
# $$
# \Vert \nabla d^{j,l} - \nabla  d^{j,l-1} \Vert_2 \leq \max \{ \Theta_{abs}, \Theta_{rel} \Vert \nabla  d^{j,l} \Vert_2 \}
# \, .
# $$

# %%
t = 0.0
T = 1.0

a_tol       = 1e-6
r_tol       = 1e-5
max_iters   = 100

while t < T:
    t += k 

    print(f"Time Step: {t}.")

    # initializing fixed point iteration
    fp_err_d    = np.inf
    val_d       = 0
    fp_iter     = 0
    
    while (not fp_err_d<= np.maximum(a_tol, r_tol * val_d )) and (fp_iter < max_iters):
        
        assemble_and_solve(d, q, ksp)

        fp_iter += 1
        val_d       = ( 2 * energy(dl) )**0.5
        fp_err_d    = ( 2 * energy(dl - d) )**0.5

        # update
        dl.x.array[:] = d.x.array[:] # NOTE that we don't update ql as it does not show up in the variational formulation

    converged = fp_err_d<= np.maximum(a_tol, r_tol * val_d )
    print(f"Fixed Point Solver took {fp_iter} iterations and converged = {converged}.")
    assert(converged) 

    # When Fixed Point Scheme is finished
    e_ela = energy(d)
    print(f"Elastic Energy: {e_ela}.")

    # update director field
    d0.x.array[:] = d.x.array[:] # NOTE that we don't update q0 as it does not show up in the variational formulation
    dl.x.array[:] = d0.x.array[:]

    # compute agreement with unit-sphere constraint
    unit_max = np.max(np.linalg.norm(np.reshape( d0.x.array[:] , (-1, dim)), axis=1))    
    unit_min = np.min(np.linalg.norm(np.reshape( d0.x.array[:] , (-1, dim)), axis=1))
    print(f"max norm of director field (min - max): ( {unit_min} - {unit_max} )")

    vtx_d.write(t)

vtx_d.close()


