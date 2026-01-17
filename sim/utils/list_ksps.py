from petsc4py import PETSc

if __name__ == "__main__":
    print(PETSc.KSP.Type.__dict__["__doc__"])