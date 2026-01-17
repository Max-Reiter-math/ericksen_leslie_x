from petsc4py import PETSc

if __name__ == "__main__":
    print(PETSc.PC.Type.__dict__["__doc__"])
    for key in PETSc.PC.Type.__dict__.keys():
        if not key.startswith("__"):
            print(f"{key}: {PETSc.PC.Type.__dict__[key]}")