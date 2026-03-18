# %% [markdown]
# # Tutorial 4: Ginzburg-Landau Penalization Method
# In this example, we implement a Ginzburg-Landau penalization for the harmonic heat flow into the sphere. This results in a non-linear equation system, which we solve using a monolithic Newton solver as in tutorial 1.
# 
# Main contents:
# - Ginzburg-Landau penalization approach,
# - parallelization.
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
# The Ginzburg-Landau penalization with penalization parameter $\varepsilon >0$ enforces the unit-sphere constraint only in a global sense. The governing equation then reads as
# 
# $$
# \partial_t d - \gamma \Delta d + \frac{\gamma}{\varepsilon} (\vert d \vert^2 -1) d    = 0 .
# $$
# 
# We equip the system with constant-in-time Dirichlet boundary conditions,
# $$
# d(t) = d_0 \text{ on } \partial \Omega \text{ for all }t\in(0,T) \, . 
# $$
# 
# The unit-sphere constraint has now been relaxed to a double-well potential in the energy law, that the system naturally admits, i.e.
# 
# $$
# \frac{1}{2} \int_{\Omega} \vert \nabla d (t) \vert^2 \mathrm{d}x
# + \frac{1}{4\varepsilon} \int_{\Omega} (\vert d (t) \vert^2 -1)^2 \mathrm{d}x
# + \int_0^t   \int_{\Omega} \gamma \vert - \Delta d(\tau) + \frac{1}{\varepsilon} (\vert d(\tau) \vert^2 -1) d (\tau) \vert^2 \mathrm{d}x \mathrm{d}\tau 
# =
# \frac{1}{2} \int_{\Omega} \vert \nabla d (0) \vert^2 \mathrm{d}x
# + \frac{1}{4\varepsilon} \int_{\Omega} (\vert d (0) \vert^2 -1)^2 \mathrm{d}x
# \, .
# $$
# 
# ## Regarding the Parallelization
# The focus of this tutorial is also on parallelization. Therefore, it may be easier to run this tutorial on several cores in the terminal using the mpirun command, i.e. to run on 2 cores type
# ```
# mpirun -n 2 python t4-Penalization.py
# ```
# In FEniCSx parallelization is achieved in the following way: The mesh and degrees of freedom are distributed across MPI processes, so each process assembles and works only on its local part of the problem plus a small layer of ghost data from neighboring subdomains. Global vectors and matrices are then combined through MPI communication, and PETSc handles the parallel linear algebra and solvers on top of that distributed data layout. In practice, you usually write the variational problem almost exactly as in serial, while FEniCSx and PETSc take care of partitioning, communication, and parallel assembly/solution. However, manual manipulation of functions needs also manual data sharing across the processes.

# %% [markdown]
# ## Step 1: Mesh Generation
# We create a disk with a centered hole as domain, i.e. $\Omega = \{x \in \mathbb{R}^2: 1<\vert x \vert < 2\}$. For simplicity, we conduct the mesh point/cell generation on the first of the parallel processes (rank 0) only. Afterwards the domain is constructed out of the union of cells and points from all ranks / processes and distributed over all ranks.

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
    if comm.rank == 0: # NOTE that we create the mesh cells and points only on rank 0
        n_radius = np.max([2,n]) # at least points on the inner and on the outer boundary are needed
        n_angle  = np.max([8,4*n]) # at least two points per quarter are needed

        # aranging points along partition of the radius as well as the angle
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

        cells_r = np.array([top_left, bottom_left, bottom_right])
        cells_l = np.array([top_left, top_right, bottom_right])
        cells = np.hstack([cells_r , cells_l])
    else:
        pts     = np.array([[],[]])
        cells   = np.array([[],[],[]])

    status = {f"rank_{MPI.COMM_WORLD.rank}": "finished"}
    status = comm.gather(status, root=0)                    # NOTE that we need to make sure that all processes have gotten this far
    if MPI.COMM_WORLD.rank == 0: 
        print("Cell and Point creation:", status)
    
    ufl_mesh    = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))) # shape yields the geometric dimension
    domain      = create_mesh(comm, cells.T, pts.T, ufl_mesh)
    return domain

print("Rank with number ", MPI.COMM_WORLD.rank," of ", MPI.COMM_WORLD.size," total ranks available.")   # print out all available ranks
domain = donut_mesh(MPI.COMM_WORLD, 15)                                                                 # NOTE that the domain configuration is done on every process.
dim    = 2

# %% [markdown]
# ## Step 2: Initialization of function spaces and functions
# For the discrete director field globally continuous, affine-linear elements (CG1) are chosen. Its consistent approximation of the Laplacian using an auxiliary variable is also achieved using globally continuous, affine-linear elements (CG1). Recall that for the specification of a non-linear problem no _TrialFunction_ is needed.
# 
# The initialization of the finite element space allows us to output the distribution of DOFs over the different processes.

# %%
from ufl import TestFunction
from dolfinx.fem import Function, functionspace, ElementMetaData

D = functionspace(domain, ElementMetaData("Lagrange", 1 , shape=(dim,))) # FE space for the director field

d1                       = Function(D)      # unknowns and result
d0                       = Function(D)      # knowns
c                        = TestFunction(D)

dofs = {f"dofs_rank_{MPI.COMM_WORLD.rank}": D.dofmap.index_map.size_local}
dofs = MPI.COMM_WORLD.gather(dofs, root=0) # NOTE that we need to make sure that all processes have gotten this far
if MPI.COMM_WORLD.rank == 0: print("DOFs per rank:", dofs)

# %% [markdown]
# ## Step 3: Variational Formulation
# For the local conservation properties of our scheme, we once again need the mass-lumped inner product.

# %%
from ufl import Measure
dxL = Measure("dx", domain = domain, metadata = {"quadrature_rule": "vertex", "quadrature_degree": 0})  # mass lumping

# %% [markdown]
# Let $k>0$ be the time-step size and $\varepsilon >0$ the penalization parameter. For $j=1,2,...$, we solve the non-linear equation system
# $$
# \frac{1}{k} (d^j -d^{j-1}, c)_h + \gamma (\nabla d^{j-1/2}, \nabla c)_2 + 1/\varepsilon ([\vert d \vert^2 -1]^{j-1/2}d^{j-1/2},c)_h  = 0 .
# $$
# for all $c\in [CG1]^N$.

# %%
from ufl import inner, dx, grad
from dolfinx.fem.petsc import NonlinearProblem

k       = 0.01
gamma   = 1.0
eps     = 0.5

d_      = 0.5*d1 + 0.5*d0                           # midpoint discretization
d_norm_ = 0.5*inner(d1,d1)+0.5*inner(d0,d0)         # midpoint discretization [|d|^2]^{j-1/2}

eq = 1/k * inner(d1 - d0, c) *dxL          # time derivative
eq += gamma*inner(grad(d_), grad(c))*dx    # damping term
eq += 1/eps*(d_norm_-1)*inner(d_, c)*dxL   # penalization term

# %% [markdown]
# ## Step 4: Initial Values
# We only need to define the initial director field $d^0$ as there are no other variables. The initial and boundary conditions are equal to those of tutorial 3.
# 
# Since we can only interpolate at the vertices that are owned by our local process, the data of the ghost DOFs needs to be shared after the function manipulation. This is achieved by the scatter_forward method.

# %%
def bd_outside(x):
    return np.isclose(x[0]**2 + x[1]**2, 4)

def bd_inside(x):
    return np.isclose(x[0]**2 + x[1]**2, 1)

def get_d0(x: np.ndarray) -> np.ndarray:
    # x has shape (dimension, points)
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
d1.interpolate(get_d0) # is used as initial guess for d^1

d0.x.scatter_forward() # NOTE that this shares the ghost values over the different processes
d1.x.scatter_forward() # NOTE that this shares the ghost values over the different processes

# %% [markdown]
# - scatter_forward() updates ghost entries by sending the current owned values of a distributed vector to neighboring processes that store those values as ghosts. You use it after modifying a function/vector locally when later computations on other ranks need consistent read access to those ghost values, for example before assembly or postprocessing.
# 
# - scatter_backward() goes the other way: it accumulates contributions from ghost entries back onto the owning process, typically with an add operation. You use it when different ranks have written partial contributions associated with shared DOFs and those contributions must be summed onto the unique owner before continuing.

# %% [markdown]
# ## Step 5: Boundary Conditions
# The boundary condition is already defined by the _callable_ **get_d0**. We need to locate the corresponding degrees of freedom (DOFs). This can be done using a geometric description of the boundary.

# %%
from dolfinx.fem import locate_dofs_geometrical, dirichletbc


dofs_inside     = locate_dofs_geometrical(D, bd_inside)
dofs_outside    = locate_dofs_geometrical(D, bd_outside)

d_initial = Function(D)
d_initial.interpolate(get_d0)
d_initial.x.scatter_forward() # NOTE that this again shares the interpolation to ghost values

bcs = [ dirichletbc(d_initial, dofs_inside), dirichletbc(d_initial, dofs_outside) ]

# %% [markdown]
# ## Step 6: Problem and Solver Setup
# We set up the non-linear problem and its Newton solver as in tutorial 1.

# %%
from dolfinx.nls.petsc import NewtonSolver

hhf = NonlinearProblem(eq, d1, bcs=bcs)
solver = NewtonSolver(MPI.COMM_WORLD, hhf)

solver.convergence_criterion = "residual" # "residual" or "incremental"
solver.atol = 1e-6
solver.rtol = 1e-5
solver.max_it = 100
solver.error_on_nonconvergence = True
solver.report = True

# %% [markdown]
# ## Step 7: Preprocessing for the Temporal Evolution
# Setup Output Pipeline for the director field and save for $t = 0$. Due to the parallel processing, integral evaluations such as the energy are only computed on the subdomain owned by the process. Therefore they need to be summed over all processes.

# %%
from dolfinx.fem import form, assemble_scalar
from dolfinx.io import VTXWriter

vtx_d = VTXWriter(MPI.COMM_WORLD, "t4-d.bp", d0, engine="BP4")

vtx_d.write(0.0)

if MPI.COMM_WORLD.rank == 0: print("Time: 0.0.")

def energy(d):
    local_energy   = assemble_scalar(form(   0.5 *  inner(grad(d), grad(d))*dx   ))     # NOTE that this is only computed locally on the subdomain owned by the process
    global_energy  = MPI.COMM_WORLD.allreduce(local_energy, op=MPI.SUM)                 # NOTE that this sums over the different parallel processes
    return global_energy

e_ela = energy(d0)
if MPI.COMM_WORLD.rank == 0: print(f"Elastic Energy: {e_ela}.")

# %% [markdown]
# For this setting, the exact stationary solution is known and given in terms of its angle between the director field and the radial direction,
# $$
# \psi (x) = \frac{\pi}{2} \frac{\log (\vert x\vert)}{\log (2)} \, .
# $$
# Consequently, we define the exact angle of the solution and prepare the error computation.

# %%
from dolfinx.fem import ElementMetaData

CG          = functionspace(domain, ElementMetaData("Lagrange", 3)) # NOTE that we increase the polynomial order to compute the error more exact
psi_exact   = Function(CG)
psi         = Function(CG)

def exact_solution(x: np.ndarray) -> np.ndarray:
    # x has shape (dimension, points)
    r       = np.sqrt(x[0]**2 + x[1]**2)   
    angles  = np.pi/2 * np.log(r)/np.log(2)
    return angles

psi_exact.interpolate(exact_solution)
psi_exact.x.scatter_forward()           # NOTE that this shares the interpolation values with the ghost dofs

# %% [markdown]
# Analogous to the energy computation, the parallel processing also affects the error computation. Integral evaluations need to be summed over all processes, whereas maximum and minimum evaluations need to be ordered over all ranks.

# %%
from ufl import SpatialCoordinate, acos, sqrt, conditional
from dolfinx.fem import Expression

coord = SpatialCoordinate(domain)

def compute_error(d):
    """
    The angle $\theta$ between two vectors $v$ and $w$ can be computed as
    cos theta = (v,w)/(|v||w|)
    """
    cos_theta       = (coord[0]*d[0] + coord[1]*d[1])/(sqrt(coord[0]**2 + coord[1]**2)*sqrt(d[0]**2 + d[1]**2))
    cos_theta_safe  = conditional(cos_theta < 1.0, cos_theta , 1.0)
    theta_safe      = acos(cos_theta_safe)    
    expr = Expression(
        theta_safe,
        CG.element.interpolation_points()
    )
    psi.interpolate(expr)
    psi.x.scatter_forward() # NOTE that this shares the interpolation values with the ghost dofs

    local_err   = assemble_scalar(form(inner(psi-psi_exact,psi-psi_exact)*dx))
    global_err  = MPI.COMM_WORLD.allreduce(local_err, op=MPI.SUM)               # NOTE that this sums the local error over all parallel procceses
    return global_err

err0 = compute_error(d0)
if MPI.COMM_WORLD.rank == 0: print(f"Initial Squared L2 Error: {err0}.")


unit_max = np.max(np.linalg.norm(np.reshape( d0.x.array[:] , (-1, dim)), axis=1))  
unit_min = np.min(np.linalg.norm(np.reshape( d0.x.array[:] , (-1, dim)), axis=1)) 
unit_max = MPI.COMM_WORLD.allreduce(unit_max, op=MPI.MAX) # NOTE that this takes the maximum over all parallel procceses
unit_min = MPI.COMM_WORLD.allreduce(unit_min, op=MPI.MIN) # NOTE that this takes the minimum over all parallel procceses
if MPI.COMM_WORLD.rank == 0: print(f"max norm of director field (min - max) combined all ranks: ( {unit_min} - {unit_max} )")

# %% [markdown]
# ## Step 8: Time evolution
# In every time step, we first solve the equation system and then update the resulting director field. Note that we need to share the updates over all processes and that printing is only done on one rank in order to not print everything multiple times.

# %%
t = 0.0
T = 1.0

while t < T:
    t += k 

    if MPI.COMM_WORLD.rank == 0: print(f"Time Step: {t}.")
        
    # solve equation system
    newton_iterations, converged = solver.solve(d1)
    d1.x.scatter_forward()

    if MPI.COMM_WORLD.rank == 0:
        print(f"Newton Solver - Converged: {converged}. Iterations: {newton_iterations}.")
    assert (converged)

    e_ela = energy(d1)                                                                      # NOTE : PARALLEL PROCESSING DONE MANUALLY
    if MPI.COMM_WORLD.rank == 0: print(f"Elastic Energy: {e_ela}.")
    
    unit_max = np.max(np.linalg.norm(np.reshape( d1.x.array[:] , (-1, dim)), axis=1))  
    unit_min = np.min(np.linalg.norm(np.reshape( d1.x.array[:] , (-1, dim)), axis=1)) 
    unit_max = MPI.COMM_WORLD.allreduce(unit_max, op=MPI.MAX)                               # NOTE : PARALLEL PROCESSING DONE MANUALLY
    unit_min = MPI.COMM_WORLD.allreduce(unit_min, op=MPI.MIN)                               # NOTE : PARALLEL PROCESSING DONE MANUALLY
    if MPI.COMM_WORLD.rank == 0: print(f"max norm of director field (min - max) combined all ranks: ( {unit_min} - {unit_max} )")


    # Error computation
    err = compute_error(d1)                                                                 # NOTE : PARALLEL PROCESSING DONE MANUALLY
    if MPI.COMM_WORLD.rank == 0:
        print(f"Squared L2 Error: {err}.\n")

    # update and save director field
    d0.x.array[:] = d1.x.array[:]
    d0.x.scatter_forward()                                                                  # NOTE : PARALLEL PROCESSING DONE MANUALLY
    
    vtx_d.write(t)

vtx_d.close()

# %% [markdown]
# Compare the computed result with the projection method! The projection method achieved for the same mesh width:
# ```
# Time Step: 1.0000000000000007.
# Before Normalization - Elastic Energy: 13.363318483144; max norm of director field (min - max): ( 0.9999999999999999 - 1.0000000001074334 )
# After Normalization - Elastic Energy: 13.363318481696894; max norm of director field (min - max): ( 0.9999999999999999 - 1.0000000000000002 )
# Squared L2 Error: 2.779664358405537e-05.
# ```
# ### This shows the decrease in accuracy using a global relaxation of the unit-sphere constraint.

# %% [markdown]
# 


