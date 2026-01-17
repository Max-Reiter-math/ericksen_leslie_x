
from mpi4py import MPI
import numpy as np
from dolfinx.fem import Function, form, assemble_scalar
from ufl import dx, grad, inner

def L2_norm(comm,f, print_local=False):
    L2_integral = inner(f , f)*dx
    err_local   = assemble_scalar(form(L2_integral))
    if print_local: print(f"Rank {comm.rank}: Local L^2-integral: {err_local}")
    err_global = comm.allreduce(err_local, op=MPI.SUM)**0.5
    return err_global

def H10_norm(comm,f, print_local=False):
    H10_integral = inner(grad(f) , grad(f))*dx
    err_local   = assemble_scalar(form(H10_integral))
    if print_local: print(f"Rank {comm.rank}: Local H^1_0-integral: {err_local}")
    err_global = comm.allreduce(err_local, op=MPI.SUM)**0.5
    return err_global

def H1_norm(comm,f, print_local=False):
    err_L2 = L2_norm(comm, f, print_local = print_local)
    err_H10 = H10_norm(comm, f, print_local = print_local)
    err_global = (err_L2**2 + err_H10**2)**0.5
    return err_global

def Linfty_norm(comm,f: Function, print_local=False):
    err_local   = np.max(np.abs(f.x.array[:]))
    if print_local: print(f"Rank {comm.rank}: Local L^infty norm: {err_local}")
    err_global = comm.allreduce(err_local, op=MPI.MAX) #NOTE - Here we do not sum over the ranks, but take the maximum
    return err_global


if __name__ == "__main__":
    from dolfinx.mesh import create_interval
    from dolfinx.fem import FunctionSpace

    """
    Test case:
    Compute norms for x, x^2, x^3, ..., x^p

    08/08/2024: yields correct results for arbitrary amount of ranks
    """

    nx = 100 # degree of fineness
    p_max = 4   # max polynomial order
    decimals = 8 # decimal points up to which the exact result is asserted
    print_local = False # whether to print local results

    comm = MPI.COMM_WORLD
    rank = comm.rank
    if rank == 0: print(f"{comm.size} MPI ranks activated")
    domain = create_interval(comm, nx, [-1,1])


    for p in range(1,p_max+1,1):

        Pp = FunctionSpace(domain, ("Lagrange", p))
        if rank == 0: 
            print(f"--- p={p} ---")
            print(f"Polynomial order of FEM space: {p}")
        comm.Barrier()

        print(f"Rank {rank}: Global dofmap size: {Pp.dofmap.index_map.size_global}")
        print(f"Rank {rank}: Local dofmap size: {Pp.dofmap.index_map.size_local}")
        print(f"Rank {rank}: Ghosts: {Pp.dofmap.index_map.ghosts}")

        f = Function(Pp)
        f.interpolate(lambda x: x[0]**p) # interpolate x^p (coordinates are always given in 3D)
        if rank == 0: print(f"Function to interpolate: x^{p}")


        print("--- L^2 ---")

        # L^2 norm
        err = L2_norm(comm,f, print_local=print_local)
        exact = ( (1 - (-1)**(2*p+1))/(2*p+1) )**0.5
        comm.Barrier()
        if rank == 0: 
            print(f"Rank {rank}: Global L^2-norm: {err}")
            print(f"Rank {rank}: Expected result: {exact}")
        assert np.round(err - exact,decimals) == 0.0

        print("--- H^1_0 ---")

        # H^1_0 norm
        err = H10_norm(comm,f, print_local=print_local)
        exact = ( (p**2)/(2*p-1)*(1 - (-1)**(2*p-1)) )**0.5
        comm.Barrier()
        if rank == 0: 
            print(f"Rank {rank}: Global H^1_0-norm: {err}")
            print(f"Rank {rank}: Expected result: {exact}")
        assert np.round(err - exact,decimals) == 0.0

        print("--- L^infty ---")

        # L^infty norm
        err = Linfty_norm(comm,f, print_local=print_local)
        comm.Barrier()
        if rank == 0: 
            print(f"Rank {rank}: Global Linfty-norm: {err}")        
            print(f"Rank {rank}: Expected result: {exact}")
        assert np.round(err - 1.0,decimals) == 0.0

        print("---")
    
