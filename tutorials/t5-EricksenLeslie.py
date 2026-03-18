# %% [markdown]
# # Tutorial 5: Ericksen-Leslie Equations
# In this example, we implement a projection method presented in **[Maximilian E. V. Reiter. (2025). Projection Methods in the Context of Nematic Crystal Flow](https://arxiv.org/abs/2502.08571)** for the simplified Ericksen-Leslie equations. This tutorial is based on the repo [ericksen_leslie_x](github.com/Max-Reiter-math/ericksen_leslie_x).
# 
# Main contents:
# - implementation of incompressible fluid models,
# - $L^2$-projection onto finite element spaces,
# - solver tuning via fieldsplit.
# 
# On the domain $\Omega \subset \mathbb{R}^{N}$, we denote the velocity field by $ v: [0,T] \times \overline{\Omega} \to \mathbb{R}^{N}$, the pressure by $P : [0,T] \times \overline{\Omega} \to \mathbb{R}$ and the director field (local average of the molecules' orientation) by $ d: [0,T] \times \overline{\Omega} \to \mathbb{R}^{N}$.
# 
# $$
# \begin{aligned}
# \partial_t v - \mu \Delta v + (v \cdot \nabla ) v + \nabla P + [\nabla d]^T (I- d \otimes d ) \Delta d & = 0 , \\
# \nabla \cdot v & = 0 , \\
# \partial_t d + (I - d \otimes d) \left [ (v \cdot \nabla) d - \gamma \Delta d \right ] & = 0 , \\
# \vert d \vert & = 1 .
# \end{aligned}
# $$
# 
# We equip the system with constant-in-time Dirichlet boundary conditions,
# $$
# v(t) = 0 \text{ on } \partial \Omega \text{ for all }t\in(0,T)\, , \qquad d(t) = d_0 \text{ on } \partial \Omega \text{ for all }t\in(0,T) \, . 
# $$

# %% [markdown]
# ## Step 1: Mesh Generation
# We create the domain $\Omega = (-1,1)^2$. Note that the _DiagonalType_ is relevant to ensure that each triangle has at least one vertex that is not on the boundary, which is a prerequisite for the inf-sup condition of P2-P1 Taylor-Hood finite elements.

# %%
import numpy as np
from dolfinx.mesh import create_rectangle, CellType
from dolfinx.cpp.mesh import DiagonalType
from mpi4py import MPI

dim = 2
n   = 5

domain = create_rectangle(MPI.COMM_WORLD, [np.array([-1.0, -1.0]), np.array([1.0, 1.0])],  [n, n], cell_type = CellType.triangle, diagonal=DiagonalType.left_right)

# %% [markdown]
# ## Step 2: Initialization of function spaces and functions
# For the velocity field and pressure, we choose the quadratic-linear P2-P1 Taylor-Hood finite element pair as it is known to be inf-sup stable. For the discrete director field globally continuous, affine-linear elements (CG1) are chosen. Its consistent approximation of the Laplacian using an auxiliary variable is also achieved using globally continuous, affine-linear elements (CG1). This can be implemented in FEniCSx using a mixed method. Although one could assemble the linearized system on a mixed product space, in practice it is often more convenient and robust to work with separate blocks for $d$ and $q$, especially when configuring block solvers and boundary conditions. In this tutorial the focus will moreover lie on the fine tuning of a nested solver.

# %%
from ufl import TrialFunction, TestFunction
from dolfinx.fem import functionspace, ElementMetaData, Function

P2  = functionspace(domain, ElementMetaData("Lagrange", 2 , shape=(dim,))) 
P1  = functionspace(domain, ("Lagrange", 1) )
D   = functionspace(domain, ElementMetaData("Lagrange", 1 , shape=(dim,)))
Y   = functionspace(domain, ElementMetaData("Lagrange", 1 , shape=(dim,)))

v1, p1, d1, q1  = TrialFunction(P2), TrialFunction(P1), TrialFunction(D), TrialFunction(Y)
v0, p0, d0, q0  = Function(P2),      Function(P1),      Function(D),      Function(Y) 
v,  p,  d,  q   = Function(P2),      Function(P1),      Function(D),      Function(Y)
a,  h,  c,  b   = TestFunction(P2),  TestFunction(P1),  TestFunction(D),  TestFunction(Y)

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
# However, the gradient of the director field in the convection term and Ericksen stress tensor are piecewise-constant and therefore not continuous. To be able to apply mass-lumping nevertheless, we apply a standard $L^2$ projection $\mathcal{P}$ onto the space $[CG1]^N$.

# %%
from ufl import Measure, inner, grad, dx
from dolfinx.fem import ElementMetaData, form
from dolfinx.fem.petsc import LinearProblem

dxL = Measure("dx", domain = domain, metadata = {"quadrature_rule": "vertex", "quadrature_degree": 0})  # mass lumping

TensorF = functionspace(domain, ElementMetaData("Lagrange", 1 , shape=(dim, dim)))                      # matrix function space for projected gradient
grad_d0 = Function(TensorF)                                                                             # gradient of d0 projected onto P1

def project_grad(d: Function, grad_d: Function, V: functionspace, bcs: list = [])-> Function:
    """
    Mass-Lumping adjusted L^2 projection of f onto the space V, i.e. this method returns Pf as solution of the equation system:
    (Pf,phi)_h = (f,phi)_2 for all phi \in V.
    """
    u = TrialFunction(V)
    v = TestFunction(V)
    # Mass Lumping
    ah = form(inner(u,v)*dxL)
    L  = form(inner(grad(d),v)*dx)
    problem = LinearProblem(ah, L,  bcs=bcs, u=grad_d) 
    problem.solve()
    return grad_d

# %% [markdown]
# Let $k>0$ be the time-step size. For $j=1,2,...$, we **first** solve a fully linear system given by
# $$
# \begin{aligned}
# \frac{1}{k}( v^j - v^{j-1}, a)_2 + \mu (\nabla v^j, \nabla a)_2 + ((v^{j-1} \cdot \nabla ) v^j,a)_2 + \frac{1}{2}((\nabla \cdot v^{j-1}) v^j, a)_2 - (P^j, \nabla \cdot a)_2 &
# \\
# - \left([\mathcal{P}\nabla d^{j-1}]^T (I - d^{j-1} \otimes d^{j-1}) q^{j}, a\right)_h & = 0 , \\
# (\nabla \cdot v^j,h)_2 & = 0 , \\
# \frac{1}{k} (d^j -d^{j-1}, c)_h + \left((I - d^{j-1} \otimes d^{j-1}) \left [ [\mathcal{P}\nabla d^{j-1}] v^j + \gamma q^j \right ] ,c\right)_h & = 0 , \\
# (q^{j}, b)_h - (\nabla d^{j}, \nabla b)_2 &= 0 \, ,
# \end{aligned}
# $$
# for all $(a,h,c,b)\in [CG2]^N \times CG1 \times [CG1]^N\times [CG1]^N$.
# 
# In the **second** step, we normalize the solution at every node $z \in \mathcal{N}$ by
# $$
# d^j (z) \leftarrow \frac{d^j (z)}{\vert d^j (z) \vert} \, .
# $$
# 
# Since we do not use a monolithic approach for the linear problem, we have to define each block in the linear system individually. Schematically, the corresponding block operator has the form
# 
# $$
# \tiny
# \begin{pmatrix}
# \frac{1}{k} (\cdot,a)_2
# + \mu (\nabla \cdot, \nabla a)_2 + ((v^{j-1} \cdot \nabla ) \cdot,a)_2 + \frac{1}{2}((\nabla \cdot v^{j-1}) \cdot, a)_2 
# &
# - (\cdot, \nabla \cdot a)_2
# &
# 0
# &
# - \left([\mathcal{P}\nabla d^{j-1}]^T (I - d^{j-1} \otimes d^{j-1}) \cdot, a\right)_h 
# \\
# -(\nabla^T \cdot,b)_2
# & 
# 0
# &
# 0
# &
# 0
# \\
# \left(  [\mathcal{P}\nabla d^{j-1}] \cdot  , (I - d^{j-1} \otimes d^{j-1}) c\right)_h 
# &
# 0
# &
# \frac{1}{k} (\cdot,c)_h
# &
# +\gamma \left((I  - d^{j-1} \otimes d^{j-1})   \cdot ,c \right)_h
# \\
# 0
# &
# 0
# &
# - (\nabla \cdot, \nabla b)_2
# &
# +(\cdot, b)_h
# \end{pmatrix}
# \cdot
# \begin{pmatrix}
# v^j\\
# P^j\\
# d^{j}\\
# q^{j}
# \end{pmatrix}
# =
# \begin{pmatrix}
# \frac{1}{k} (v^{j-1},a)_2
# \\
# 0
# \\
# \frac{1}{k} (d^{j-1},c)_h
# \\
# 0
# \end{pmatrix}
# \, .
# $$

# %%
from petsc4py import PETSc
from ufl import inner, dx, grad, div, dot, nabla_grad
from dolfinx.fem import form, Constant

k       = 0.01     # time-step size
mu      = 1.0       # diffusion parameter
gamma   = 1.0       # damping parameter

# momentum equation
a11  =  inner(v1, a )*dx                                                         # discrete time derivative
a11 +=  k* ( inner(dot(v0, nabla_grad(v1)), a) + 0.5*div(v0)*inner(v1, a)  )*dx  # skew-symmetric convection term (see Temam)
a11 +=  k*mu*inner( grad(v1), grad(a))*dx                                        # diffusion term
a12  = -k*inner(p1, div(a)) * dx                                                 # pressure term
a14  = -k*inner(dot(grad_d0 , a), q1)*dxL                                        # Ericksen stress
a14 +=  k*inner(dot(grad_d0 , a), d0)*inner(d0, q1)*dxL                          # Ericksen stress
L1   =  inner(v0, a )*dx                                                         # discrete time derivative
# incompressibility constraint
a21  = -k*inner(div(v1), h)*dx  # incompressibility constraint
z    = Function(P1)
a22  = z*p1*h*dx                # mass matrix multiplied by zero for computational reasons
L2   = Constant(domain, PETSc.ScalarType(0))*h*dx
# director equation
a33  = inner(d1, c)*dxL                                         # discrete time derivative
L3   = inner(d0, c)*dxL                                         # discrete time derivative
a31  = k*gamma *inner(dot(grad_d0 , v1), c)*dxL                 # convective term
a31 -= k*gamma *inner(dot(grad_d0 , v1), d0)*inner(d0, c)*dxL   # convective term
a34  = k*gamma *inner(q1, c)*dxL                                # damping term
a34 -= k*gamma *inner(q1, d0)*inner(d0, c)*dxL                  # damping term
# equation for the variational derivative / Laplacian 
a43  = -inner(grad(d1),grad(b))*dx
a44  = inner(q1, b)*dxL
L4   = inner(Constant(domain, (PETSc.ScalarType(0), PETSc.ScalarType(0))), b)*dx

ah  = form([
        [a11,  a12,  None, a14], 
        [a21,  a22,  None, None],
        [a31,  None, a33,  a34], 
        [None, None, a43,  a44]
    ])

L   = form([
        L1 ,
        L2 ,
        L3 ,
        L4
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
# We only need to define the initial director field $d^0$ as the initial Laplacian is not used in the variational formulation of the discretization and as we set the initial velocity field to be zero. However, the gradient of the initial director field needs a projection onto the space $CG1$.

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

d0.interpolate(get_d0)

project_grad(d0, grad_d0, TensorF, bcs=[])  # project gradient of d0 onto CG1 space

# %% [markdown]
# ## Step 5: Boundary Conditions
# As prior, we locate the DOFs geometrically. 

# %%
from dolfinx.fem import locate_dofs_geometrical, dirichletbc

def boundary_2d(x: np.ndarray) -> np.ndarray:
    return np.logical_or.reduce((np.isclose(x[0], -1.0), np.isclose(x[0], 1.0), np.isclose(x[1], -1.0), np.isclose(x[1], 1.0)))

dofs_V = locate_dofs_geometrical(P2, boundary_2d)
dofs_D = locate_dofs_geometrical(D, boundary_2d)

v_bc = Function(P2)         # no-slip boundary conditions
d_bc = Function(D)         
d_bc.interpolate(get_d0)    # constant in time Dirichlet boundary conditions

bcs = [
        dirichletbc(v_bc, dofs_V), 
        dirichletbc(d_bc, dofs_D)
    ]

# %% [markdown]
# ## Step 6: Assembly and Setup
# We set up the linear problem and its solver by assembling the matrix and right-hand side vector. Thereby, we have to set a null space for the pressure as the pressure is only defined up to a constant without Dirichlet boundary conditions.

# %%
from petsc4py import PETSc
import dolfinx.la as la
from dolfinx.fem import Function, bcs_by_block, extract_function_spaces, bcs_by_block
from dolfinx.fem.petsc import assemble_matrix_nest, assemble_vector_nest, apply_lifting_nest, set_bc_nest, set_bc_nest, create_vector_nest

def assemble_all(a, L, bcs = []):
    # Assemble nested matrix operators
    # NOTE - the following is based on the dolfinx tutorials        
    A = assemble_matrix_nest(a, bcs=bcs)
    A.assemble()

    # Assemble right-hand side vector
    b = assemble_vector_nest(L)
    
    apply_lifting_nest(b, a, bcs=bcs)                       # modifies the assembled right-hand side to account for the effect of Dirichlet conditions that have been imposed on the matrix blocks, so the linear system remains algebraically consistent
    bcs0 = bcs_by_block(extract_function_spaces(L), bcs)    # reorganizes the boundary conditions by block/function space,
    set_bc_nest(b, bcs0)                                    # inserts the prescribed boundary values into the corresponding entries of the nested right-hand side vector.

    # Since the variational formulation includes only the pressure gradient, the pressure is only fixed up to a constant, unless Dirichlet boundary conditions for the pressure are prescribed
    null_vec = create_vector_nest(L)                    # Create a nested PETSc vector with the same block structure/layout as the linear form L.
    null_vecs = null_vec.getNestSubVecs()               # Extract the individual sub-vectors (one per block) from the nested vector.
    null_vecs[0].set(0.0)                               # Set all entries in the first block to zero.
    null_vecs[1].set(1.0)                               # Set all entries in the second block to one, defining the candidate nullspace direction.
    null_vec.normalize()                                # Normalize the full nested vector to unit length.
    nsp = PETSc.NullSpace().create(vectors=[null_vec])  # Build a PETSc nullspace object from this vector.
    assert nsp.test(A)                                  # Verify that the matrix A annihilates this vector, i.e. that it is truly in the nullspace.
    A.setNullSpace(nsp)                                 # Attach the nullspace to A so PETSc solvers know about the singular direction.

    return (A, b)

(A, b) = assemble_all(ah, L, bcs = bcs)

# %% [markdown]
# ## Step 7: Solver Setup and Tuning
# To solve the large linear system efficiently, we do not use a direct solve on the full coupled matrix, but an iterative Krylov method together with a preconditioner. A preconditioner is an auxiliary matrix or operator that approximates the original system in a form that is much easier to invert; it is applied inside each iteration to improve the conditioning of the problem and thereby reduce the number of iterations needed for convergence. In block-structured multiphysics problems, a good preconditioner is often built from simpler approximations of the diagonal blocks, since these capture the main physics of each variable while remaining cheap to solve.
# 
# In the present case, the system matrix has a nested block structure corresponding to the unknowns $u,p,d,q$. We therefore construct a block preconditioner $P$ that reuses selected diagonal blocks of the full matrix $A$ and replaces others by simpler assembled approximations. This block preconditioner is then combined with a PETSc fieldsplit strategy, which treats the different variable blocks separately and assigns a tailored subsolver or subpreconditioner to each field. This is a standard and effective approach for coupled finite element systems, because different variables often have very different algebraic properties and benefit from different solver choices.

# %%
from dolfinx.fem.petsc import assemble_matrix

def get_preconditioner(A):
    # Create a nested matrix P to use as the preconditioner. The
    # top-left block of P is shared with the top-left block of A. The
    # bottom-right diagonal entry is assembled from the form a_p11:
    P11 = assemble_matrix(form(inner(p1, h) * dx), bcs=[])
    P = PETSc.Mat().createNest([
        [A.getNestSubMatrix(0, 0), None, None, None], 
        [None, P11, None, None],
        [None, None, A.getNestSubMatrix(2, 2), None],
        [None, None, None, A.getNestSubMatrix(3, 3)],
        ])
    P.assemble()

    A00 = A.getNestSubMatrix(0, 0)                      # Extract the (0,0) block of the nested system matrix A, here the velocity block.
    A00.setOption(PETSc.Mat.Option.SPD, True)           # Tell PETSc to treat this block as symmetric positive definite, so it may use algorithms/preconditioners that exploit this structure.
    P00, P11 = P.getNestSubMatrix(0, 0), P.getNestSubMatrix(1, 1)  # Extract the first and second diagonal blocks of the preconditioner matrix P.
    P00.setOption(PETSc.Mat.Option.SPD, True)          # Mark the first preconditioner block as symmetric positive definite.
    P11.setOption(PETSc.Mat.Option.SPD, True)          # Mark the second preconditioner block as symmetric positive definite.

    return P

def configure_solver(ksp, A, P):   

    ksp.setOperators(A, P)
    
    ksp.setType("gmres")
    ksp.setTolerances(rtol=1e-9)
    ksp.getPC().setFactorSolverType(PETSc.Mat.SolverType.MUMPS)
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)

    # Return the index sets representing the row and column spaces. 2 times Blocks spaces
    nested_IS = P.getNestISs()
    ksp.getPC().setFieldSplitIS(("u", nested_IS[0][0]), ("p", nested_IS[0][1]), ("d", nested_IS[0][2]), ("q", nested_IS[0][3]))

    # Set the preconditioners for each block
    ksp_u, ksp_p, ksp_d, ksp_q = ksp.getPC().getFieldSplitSubKSP()
    ksp_u.setType("preonly") 
    ksp_u.getPC().setType("hypre") 
    ksp_p.setType("preonly") 
    ksp_p.getPC().setType("jacobi") 
    ksp_d.setType("preonly") 
    ksp_d.getPC().setType("jacobi") 
    ksp_q.setType("preonly") 
    ksp_q.getPC().setType("sor") 

P = get_preconditioner(A)
solver = PETSc.KSP().create(MPI.COMM_WORLD)
configure_solver(solver, A, P)

# %% [markdown]
# There are many more options for [Krylov Solvers](https://petsc.org/main/manual/ksp/#tab-kspdefaults) and [Preconditioners](https://petsc.org/main/manual/ksp/#tab-pcdefaults).

# %% [markdown]
# ## Step 8: Preprocessing for the Temporal Evolution
# Setup Output Pipeline for the director and velocity field and save for $t = 0$. Using the VTX writer allows us to save the velocity field at every node instead of only at the triangle vertices, which would be the case in the XDMF format.

# %%
from dolfinx.fem import form, assemble_scalar
from dolfinx.io import VTXWriter

vtx_v = VTXWriter(MPI.COMM_WORLD, "t5-v.bp", v0, engine="BP4")
vtx_d = VTXWriter(MPI.COMM_WORLD, "t5-d.bp", d0, engine="BP4")

vtx_v.write(0.0)
vtx_d.write(0.0)

print("Time: 0.0.")

def energies(v,d):
    e_kin = assemble_scalar(form(   0.5 *  inner(v, v)*dx   ))
    e_ela = assemble_scalar(form(   0.5 *  inner(grad(d), grad(d))*dx   ))
    return e_kin, e_ela

e_kin, e_ela = energies(v0, d0)
print(f"Kinetic Energy: {e_kin}; Elastic Energy: {e_ela}.")

# %% [markdown]
# ## Step 9: Time evolution
# In every time step, we first solve the equation system and then update the resulting director field.

# %%
t = 0.0
T = 1.0

x = PETSc.Vec().createNest([la.create_petsc_vector_wrap(v.x), la.create_petsc_vector_wrap(p.x), la.create_petsc_vector_wrap(d.x), la.create_petsc_vector_wrap(q.x)])

while t < T:
    t += k 

    print(f"Time Step: {t}.")
    (A, b) = assemble_all(ah, L, bcs = bcs)
    P = get_preconditioner(A)
    solver.setOperators(A, P)
    
    solver.solve(b, x)

    # properties before nodal normalization
    e_kin, e_ela = energies(v, d)
    unit_max = np.max(np.linalg.norm(np.reshape( d.x.array[:] , (-1, dim)), axis=1))    
    unit_min = np.min(np.linalg.norm(np.reshape( d.x.array[:] , (-1, dim)), axis=1))
    print(f"Before Normalization - Kinetic Energy: {e_kin}; Elastic Energy: {e_ela}; max norm of director field (min - max): ( {unit_min} - {unit_max} )")

    # Normalization
    nodal_normalization(d, dim)
    
    # properties after nodal normalization
    e_kin, e_ela = energies(v, d)
    unit_max = np.max(np.linalg.norm(np.reshape( d.x.array[:] , (-1, dim)), axis=1))    
    unit_min = np.min(np.linalg.norm(np.reshape( d.x.array[:] , (-1, dim)), axis=1))
    print(f"After Normalization - Kinetic Energy: {e_kin}; Elastic Energy: {e_ela}; max norm of director field (min - max): ( {unit_min} - {unit_max} )")
    
    # update and save director field
    v0.x.array[:] = v.x.array[:] 
    d0.x.array[:] = d.x.array[:] 
    project_grad(d0, grad_d0, TensorF, bcs = [])
    
    # NOTE that we don't update p0, q0 as they do not show up in the variational formulation
    vtx_v.write(t)
    vtx_d.write(t)

vtx_v.close()
vtx_d.close()


