from ufl import grad, div, curl, inner, dot, cross, dx, Measure, FacetNormal, TrialFunction, TestFunction
import dolfinx.la as la
from dolfinx.fem import Function, functionspace, form, assemble_scalar, ElementMetaData, bcs_by_block, extract_function_spaces, bcs_by_block
from dolfinx.fem.petsc import assemble_matrix, assemble_matrix_nest, assemble_vector_nest, apply_lifting_nest, set_bc_nest, create_vector_nest, set_bc_nest
from sim.common.operators import *
from sim.common.common_fem_methods import *
from sim.common.meta_bcs import *

def saddlept_fp(comm, experiment, args, postprocess=None):
    """
    Implicit method (see Equations 20 and 15) by Badia, S., Guillen-González, F., & Gutiérrez-Santacreu, J. (2011). Finite element approximation of nematic liquid crystal flows using a saddle-point structure. J. Comput. Phys., 230(4), 1686-1706.

    Implemented using an iterative Picard-type linearization keeping the decoupling of the Quasi-Newton-method.

    Important properties of the algorithm:
    - fulfills a discrete energy law
    - no auxiliary variable for the variational derivative of the energy needed
    - penalization variable for the unit-norm constraint
    """

    if postprocess and comm.rank == 0:
        postprocess.log("dict", "static", {"# MPI Ranks": comm.size})

    # SECTION PARAMETERS
    dim = args.dim
    dt  = args.dt
    t   = 0

    
    if postprocess and comm.rank == 0:
        postprocess.log("dict", "static",{"SID": args.sim_id, "MODEL":args.mod, "PROJECTION STEP": args.projection_step, "EXPERIMENT":args.exp, "mesh res.": args.dh, "dt":args.dt, "dim":args.dim, "T":args.T, "fp_a_tol" : args.fp_a_tol, "fp_r_tol" : args.fp_r_tol, "fp_max_iters" : args.fp_max_iters})

    #!SECTION PARAMETERS

    domain        = experiment.mesh    
    meshtags      = experiment.meshtags    
    n_F           = FacetNormal(domain)

    #add all local variables for transparency    
    if postprocess and comm.rank == 0:
        postprocess.log("dict","static",{"model.vars":dict(locals())}, visible =False)

    initial_conditions          = experiment.initial_conditions
    boundary_conditions         = experiment.boundary_conditions

    total_time_start = mpi_time(comm)
    
    # SECTION - FUNCTION SPACES AND FUNCTIONS

    P2          = functionspace(domain, ElementMetaData("Lagrange", 2 , shape=(dim,))) 
    P1          = functionspace(domain, ("Lagrange", 1) )
    D, Y        = functionspace(domain, ElementMetaData("Lagrange", 1 , shape=(dim,))), functionspace(domain, ElementMetaData("Lagrange", 1 ))
    
    v1,p1, d1, xi1                      = TrialFunction(P2),TrialFunction(P1), TrialFunction(D), TrialFunction(Y)
    v0,p0, d0, xi0                      = Function(P2),Function(P1), Function(D), Function(Y) 
    vl,pl, dl, xil                      = Function(P2),Function(P1), Function(D), Function(Y)
    vl0, dl0                            = Function(P2), Function(D) # for inner fp iteration
    a, h, c, psi                        = TestFunction(P2),TestFunction(P1), TestFunction(D), TestFunction(Y)
    
    # FOR COMPUTATION OF INITIAL q
    qc1 = TrialFunction(Y)
    q_ic = Function(Y)
    bc_test = TestFunction(Y)  

    # COMPUTE AND SAVE DOFS PER RANK
   
    local_dofs = P2.dofmap.index_map.size_local + P1.dofmap.index_map.size_local + D.dofmap.index_map.size_local + Y.dofmap.index_map.size_local    # Count DOFs on this rank    
    local_info = {f"rank {comm.rank} V-dofs": local_dofs}                                                                                           # Create local dictionary   
    all_info = comm.gather(local_info, root=0)                                                                                                      # Gather all dictionaries at root

    if comm.rank == 0:
        # Merge all into a single dict
        combined = {}
        for partial_dict in all_info:
            combined.update(partial_dict)
        postprocess.log("dict", "static", combined)

    #!SECTION FUNCTION SPACES AND FUNCTIONS
    
    # SECTION VARIATONAL FORMULATION
    dxL = Measure("dx", domain = domain, metadata = {"quadrature_rule": "vertex", "quadrature_degree": 0})

    # SECTION INITIAL CONDITIONS
    v0.interpolate(initial_conditions["v"]) 
    p0.interpolate(initial_conditions["p"]) 
    d0.interpolate(initial_conditions["d"]) 
    if "H" in initial_conditions.keys():
        H = Function(D)
        H.interpolate(initial_conditions["H"])
        # TODO - scatter forward
    else:
        H = None
    scatter_all([v0, p0, d0])
    update_and_scatter([dl0, vl0], [d0, v0])
    
    
    #!SECTION

    #SECTION - BOUNDARY CONDITIONS
    bcs1, bcs2 = [], []

    p_dirichlet_bcs_exist = False # NOTE - this becomes relevant later for the assembly and solver setup to decide whether a nullspace needs to be prescribed
    
    for bcwofs in boundary_conditions:
        if bcwofs.type == "Dirichlet":
            if bcwofs.quantity == "v":
                bcwofs.set_fs(P2)
                bcs1.append(bcwofs.bc)
            elif bcwofs.quantity == "p":
                bcwofs.set_fs(P1)
                bcs1.append(bcwofs.bc)
                p_dirichlet_bcs_exist = True
            elif bcwofs.quantity == "d":
                bcwofs.set_fs(D)
                bcs2.append(bcwofs.bc)
        # elif bcwofs.type == "Neumann":            
        # elif bcwofs.type == "Robin":
        else: postprocess.log("dict", "static",{"Warning" : "Boundary conditions of type "+bcwofs.type+" are currently not implemented and will be ignored..."} )
    
    #!SECTION

    #SECTION - VARIATIONAL FORM
    a1, L1, a2, L2 = variational_form(dxL, v1, p1, d1, xi1, vl0, dl0, v0, p0, d0, xi0, a, h, psi, c,  dt, n_F, args, H=H, f= None,  p_dirichlet_bcs_exist = p_dirichlet_bcs_exist)
    #!SECTION
    
    #SECTION - POSTPROCESSING FOR t=0

    if postprocess:
        postprocess.log_functions(0.0, {"v": v0, "p":p0, "d":d0, "xi":xi0}, mesh = domain) #, meshtags = meshtags)

    metrics = compute_metrics(comm, args, v0,d0, H=H) # for initial condition
    if postprocess and comm.rank == 0:
        postprocess.log("dict", t, { "time" : t} |  metrics )

    #!SECTION

    # SECTION TIME EVOLUTION
    total_time = mpi_time(comm, start = total_time_start )

    while t < args.T:
        t += dt 


        # INITIALIZING FIXED POINT ITERATION
        fp_err_v, fp_err_d  = np.inf, np.inf
        val_v, val_d = 0, 0
        fp_iter = 0

        assembly_time, time_solsetup, time_sol = 0, 0, 0

        #SECTION - FIXPOINT ITERATION
        while not (fp_err_v <= np.maximum(args.fp_a_tol, args.fp_r_tol * val_v ) and fp_err_d<= np.maximum(args.fp_a_tol, args.fp_r_tol * val_d )) and (fp_iter < args.fp_max_iters):
            fp_iter += 1

            # SECTION - ASSEMBLY 1
            measure_assembly_start = mpi_time(comm)
            A2, b2 = assemble_all(a2, L2, bcs = bcs2)          
            assembly_time += mpi_time(comm, start= measure_assembly_start)
            #!SECTION 

            #SECTION - SOLVER 1
            # SETUP SOLVERS
            start_solsetup = mpi_time(comm)
            ksp2 = setup_solver2(comm, args, A2)
            time_solsetup += mpi_time(comm, start = start_solsetup)

            x2 = PETSc.Vec().createNest([la.create_petsc_vector_wrap(dl.x), la.create_petsc_vector_wrap(xil.x)])

            # SOLVING
            start_sol = mpi_time(comm)
            ksp2.solve(b2, x2)
            time_sol += mpi_time(comm, start = start_sol)
            #!SECTION

            #SECTION - NODAL PROJECTION STEP AND UPDATING FP CRITERIA 1
            if args.projection_step == 1:
                nodal_normalization(dl, dim)

            fp_err_d, val_d = compute_fp_metrics_ela(comm, args, dl, dl0)    
            #!SECTION

            #SECTION - UPDATE AT END OF FIXED POINT ITERATION
            update_and_scatter([dl0], [dl])
            #!SECTION

            # SECTION - ASSEMBLY 2
            measure_assembly_start = mpi_time(comm)
            A1, b1 = assemble_all(a1, L1, bcs = bcs1, set_nullspace = not( p_dirichlet_bcs_exist ) )    
            assembly_time += mpi_time(comm, start= measure_assembly_start)
            #!SECTION 

            #SECTION - SOLVER 1
            # SETUP SOLVERS
            start_solsetup = mpi_time(comm)
            ksp1 = setup_solver1(comm, args, A1, p1, h)
            time_solsetup += mpi_time(comm, start = start_solsetup)

            x1 = PETSc.Vec().createNest([la.create_petsc_vector_wrap(vl.x), la.create_petsc_vector_wrap(pl.x)])

            # SOLVING
            start_sol = mpi_time(comm)
            ksp1.solve(b1, x1)
            time_sol += mpi_time(comm, start = start_sol)
            #!SECTION


            #SECTION - UPDATING FP CRITERIA 2 AND METRICS

            fp_err_v, val_v = compute_fp_metrics_kin(comm, args, vl, vl0)    
            
            metrics = compute_metrics(comm, args, vl,dl,H=H)
            # FP ERROR AS ABSOLUTE OR RELATIVE??
            
            if postprocess and comm.rank == 0:
                postprocess.log("dict", t, { "time" : t} | metrics , visible = True)
                postprocess.log("dict", t, {
                    "time":t,
                    "t.ass": assembly_time, 
                    "t.sol": time_sol, 
                    "t.solsetup" : time_solsetup,
                    "fp.errv" : fp_err_v,
                    "fp.errd" : fp_err_d, 
                    "fp.iters": fp_iter,
                    }, visible = True)

            #!SECTION

            #SECTION - UPDATE AT END OF FIXED POINT ITERATION
            update_and_scatter([vl0], [vl])
            #!SECTION

    
        #SECTION - UPDATE
        update_and_scatter([v0,p0,d0,xi0], [vl,pl,dl,xil])
        #!SECTION 

        
        
        
        #SECTION - METRICS AT END OF ITERATION 
        errorL2 = np.nan
        if experiment.has_exact_solution:   
            errorL2 = experiment.compute_error(comm, dl,t,norm = "L2", degree_raise = 3)   

        total_time = mpi_time(comm, start = total_time_start )
        if postprocess and comm.rank == 0:
            postprocess.log("dict", t, { "time" : t, "errorL2" : errorL2 , "t.tot" : total_time} )
        
        #!SECTION

        #SECTION - SAVING
        if postprocess: 
            postprocess.log_functions(t, {"v": v0, "p":p0, "d":d0, "xi":xi0}, mesh = domain, meshtags = meshtags)

        #!SECTION

    #!SECTION TIME EVOLUTION

    if postprocess: 
        postprocess.close()

#!SECTION GENERAL METHOD

def assemble_all(a, L, bcs = [], set_nullspace = False):
    # Assemble nested matrix operators
    A = assemble_matrix_nest(a, bcs=bcs)
    A.assemble()

    # Assemble right-hand side vector
    b = assemble_vector_nest(L)

    # Modify ('lift') the RHS for Dirichlet boundary conditions
    apply_lifting_nest(b, a, bcs=bcs)

    # Sum contributions for vector entries that are share across
    # parallel processes
    for b_sub in b.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Set Dirichlet boundary condition values in the RHS vector
    bcs0 = bcs_by_block(extract_function_spaces(L), bcs)
    set_bc_nest(b, bcs0)

    # Since the variational formulation includes only the pressure gradient, the pressure is only fixed up to a constant.
    # This is only true, unless Dirichlet boundary conditions for the pressure are prescribed
    if set_nullspace:
        # We supply a vector that spans the nullspace to the solver, and any component
        # of the solution in this direction will be eliminated during the
        # solution process.
        null_vec = create_vector_nest(L)

        # Set velocity part to zero and the pressure part to a non-zero
        # constant
        null_vecs = null_vec.getNestSubVecs()
        null_vecs[0].set(0.0), null_vecs[1].set(1.0)

        # Normalize the vector that spans the nullspace, create a nullspace
        # object, and attach it to the matrix
        null_vec.normalize()
        nsp = PETSc.NullSpace().create(vectors=[null_vec])
        assert nsp.test(A)
        A.setNullSpace(nsp)
    
    return A, b

def variational_form(dxL, v1, p1, d1, xi1, vl0, dl0, v0, p0, d0, xi0, a, h, psi, c, dt, normal_F, args, H=None, f= None, p_dirichlet_bcs_exist = False):
        """
        The variational form is structured as followed:
        a(.,.) : bilinear form describing the lhs of the system
        L(.) : linear form describing the rhs of the system
        a_ij(.,.) : bilinear form describing the lhs of the system depending on the i-th test function and the j-th trial function
        e.g.
        a_11 --> depends on v1 and a
        a_14 --> depends on x11 and a
        """

        vl0_ = 0.5*vl0 + 0.5*v0
        dl0_ = 0.5*dl0 + 0.5*d0

        # SECTION MOMENTUM EQUATION
        a_11 = inner( v1, a )*dx # discrete time derivative
        L_1  = inner( v0, a )*dx

        # Convection in Crank-Nicholson discretizaiont
        a_11 += 0.5*dt* ( inner(dot(vl0_, nabla_grad(v1)), a) + 0.5*div(vl0_)*inner(v1, a)  )*dx 
        L_1 -= 0.5*dt* ( inner(dot(vl0_, nabla_grad(vl0)), a) + 0.5*div(vl0_)*inner(vl0, a)  )*dx 

        if args.mu4 != 0.0:
            if args.sym_grad:
                a_11 += 0.5*dt * args.mu4 *inner( grad_sym(v1), grad_sym(a))*dx
                L_1  -= 0.5*dt * args.mu4 *inner( grad_sym(vl0), grad_sym(a))*dx
            else:            
                a_11 += 0.5*dt* args.mu4 *inner( grad(v1), grad(a))*dx                
                L_1  -= 0.5*dt* args.mu4 *inner( grad(vl0), grad(a))*dx


        coeff1 = args.mu1 + (args.lam)**2 /args.gamma
        if coeff1 != 0.0:
            a_11 += 0.5*dt * coeff1*inner(d0,dot(grad_sym(v1),d0))*inner(d0,dot(grad_sym(a),d0))*dx            
            L_1  -= 0.5*dt * coeff1*inner(d0,dot(grad_sym(vl0),d0))*inner(d0,dot(grad_sym(a),d0))*dx


        coeff2 = args.mu5 + args.mu6 - (args.lam)**2 /args.gamma
        if coeff2 != 0.0:
            a_11 += 0.5*dt * coeff2 *inner( dot(grad_sym(v1),d0), dot(grad_sym(a),d0))*dx
            L_1  -= 0.5*dt * coeff2 *inner( dot(grad_sym(vl0),d0), dot(grad_sym(a),d0))*dx

        
        # Reformulated pressure term
        a_12 = dt*(-1)*inner(p1, div(a)) * dx 
        if p_dirichlet_bcs_exist:
            a_12 += dt*inner(p1, dot(a, normal_F)) * ds
            # NOTE - this is not the most elegant implementation
            # On the parts of the boundary where we strongly enforce the dirichlet bcs for the pressure, this will simply turn into a prescribed forcing term during assembly.
            # On the parts where this is not the case and further no-slip bcs for the velocity are employed, this term will simply vanish during assembly. 
            # This could be made more efficient by case distinction etc.

        # Ericksen stress tensor
        # Crank-Nicholson discretization
        L_1 += 0.5*dt*q_elastic_energy(args, dl0, dl0, d0, dot(grad(d0) , a), H = H)
        L_1 += 0.5*dt*q_elastic_energy(args, d0, d0, d0, dot(grad(d0) , a), H = H)

        # Leslie stress tensor
        if args.beta != 0.0:            
            # Crank-Nicholson discretization
            L_1 -= 0.5*dt* args.beta *q_elastic_energy(args, dl0, dl0, d0, dot(grad_skw(a),dl0_), H = H)
            L_1 -= 0.5*dt* args.beta *q_elastic_energy(args, d0, d0, d0, dot(grad_skw(a),dl0_), H = H)
        if args.lam != 0.0:            
            # Crank-Nicholson discretization
            L_1 += 0.5*dt* args.lam *q_elastic_energy(args, dl0, dl0, d0, dot(grad_sym(a),dl0_), H = H)
            L_1 += 0.5*dt* args.lam *q_elastic_energy(args, d0, d0, d0, dot(grad_sym(a),dl0_), H = H)


        if f is not None:
            L_1 += dt*inner(f, a)*dx

        #!SECTION MOMENTUM EQUATION

        # SECTION DIVERGENCE ZERO CONDITION
        # see below directly in definition of a
        #!SECTION DIVERGENCE ZERO CONDITION
        
        zero_h = Function(p0.function_space)
        L_2 = inner(zero_h, h) * dx

        #!SECTION DIVERGENCE ZERO CONDITION

        # SECTION DIRECTOR EQUATION
        a_33 = inner(d1, c)*dx
        L_3  = inner(d0, c)*dx

        # Ericksen stress tensor
        L_3 -= dt*inner(dot(grad(d0) , vl0_),  c)*dx

        # Leslie stress tensor
        if args.beta != 0.0:
            a_33 -= 0.5*dt* args.beta *inner(dot(grad_skw(vl0_),d1), c)*dx
            L_3  += 0.5*dt* args.beta *inner(dot(grad_skw(vl0_),d0), c)*dx
        if args.lam != 0.0:
            a_33 += 0.5*dt* args.lam *inner(dot(grad_sym(vl0_),d1), c)*dx            
            L_3  += 0.5*dt* args.lam *inner(dot(grad_sym(vl0_),d0), c)*dx


        if args.gamma != 0.0:
            # Crank-Nicholson discretization
            a_33 += 0.5*dt* args.gamma * q_elastic_energy(args, d1, dl0, d0, c, H = H)
            L_3  -= 0.5*dt* args.gamma * q_elastic_energy(args, d0, d0, d0, c, H = H)

        a_34 = dt*args.gamma*xi1*inner(d0,c)*dxL

        #!SECTION DIRECTOR EQUATION

        # SECTION EQUATION FOR THE VARIATIONAL DERIVATIVE
        # TODO - recheck correct formulation here for full oseen-Frank energy
        a_43 = psi*inner(d1,dl0)*dxL

        a_44 = (-1)*(args.alpha**2)* xi1 * psi *dxL

        L_4 = 1.0 * psi *dxL

        #!SECTION EQUATION FOR THE VARIATIONAL DERIVATIVE
        
        """
        Full bilinear form (lhs) and linear form (rhs)
        """
        zero_block_p =  zero_h * p1 * h * dx if p_dirichlet_bcs_exist else None # NOTE - the zero block is assembled in the case of Dirichlet boundary conditions for the pressure
        a1 = form([
            [a_11, a_12],
            [(-1)*inner(div(v1), h) * dx, zero_block_p],
        ])

        L1 = form([
            L_1 ,
            L_2 ,
        ])
        
        a2 = form([
            [ a_33, a_34],
            [ a_43, a_44]
        ])

        L2 = form([
            L_3 ,
            L_4 ,
        ])

        return a1, L1, a2, L2

def q_elastic_energy(args, d1, d_, d0, b, H = None):
    eq = args.K1 * inner( grad(d1), grad(b))*dx
    if args.K2 != 0.0:
        eq += args.K2 * inner( div(d1), div(b))*dx
    if args.K3 != 0.0:
        eq += args.K3 * inner( curl(d1), curl(b))*dx
    if args.K4 != 0.0:
        eq += args.K4 * inner( d_, curl(d1)) * inner(d0, curl(b)) *dx
        eq += args.K4 * inner( d_, curl(d_)) * inner(b, curl(d1)) *dx
        #NOTE - Since the curl is present a simplification to 2D is not possible.
    if args.K5 != 0.0:
        eq += args.K5 * inner( cross( d_, curl(d1)) , cross(d0, curl(b)) )*dx
        eq += args.K5 * inner( cross( d_, curl(d_)) , cross(b, curl(d1)) )*dx
        #NOTE - The cross product could be replaced by an according tangential matrix. However, since the curl is present a simplification to 2D is not possible anyways.

    if H != None and args.chi_vert != 0.0:
        eq -= args.chi_vert * inner( d1, H)*inner( b, H)*dx
    if H != None and args.chi_perp != 0.0:
        eq += args.chi_perp * inner(  H, a_times_b_times_c(d1, b, H) )*dx

    return eq

def setup_solver1(comm, args, A, p1, h):
    # Create a nested matrix P to use as the preconditioner. The
    # top-left block of P is shared with the top-left block of A. The
    # bottom-right diagonal entry is assembled from the form a_p11:
    P11 = assemble_matrix(form(inner(p1, h) * dx), bcs=[])
    P = PETSc.Mat().createNest([
        [A.getNestSubMatrix(0, 0), None], 
        [None, P11],
        ])
    P.assemble()


    A00 = A.getNestSubMatrix(0, 0)
    A00.setOption(PETSc.Mat.Option.SPD, True)

    P00, P11 = P.getNestSubMatrix(0, 0), P.getNestSubMatrix(1, 1)
    P00.setOption(PETSc.Mat.Option.SPD, True)
    P11.setOption(PETSc.Mat.Option.SPD, True)


    # Create a MINRES Krylov solver and a block-diagonal preconditioner
    # using PETSc's additive fieldsplit preconditioner
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A, P)
    
    
    # ksp.setOperators(A_ass) #, A_ass)
    ksp.setType("gmres")
    ksp.setTolerances(rtol=1e-9)
    ksp.getPC().setFactorSolverType(PETSc.Mat.SolverType.MUMPS)
    # """
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)

    

    # Define the matrix blocks in the preconditioner with the velocity
    # and pressure matrix index sets

    # Return the index sets representing the row and column spaces. 2 times Blocks spaces
    nested_IS = P.getNestISs()
    ksp.getPC().setFieldSplitIS(("u", nested_IS[0][0]), ("p", nested_IS[0][1]))

    # Set the preconditioners for each block. For the top-left
    # Laplace-type operator we use algebraic multigrid. For the
    # lower-right block we use a Jacobi preconditioner. By default, GAMG
    # will infer the correct near-nullspace from the matrix block size.
    ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
    ksp_u.setType(args.ksp_type_u) 
    ksp_u.getPC().setType(args.pc_type_u) 
    ksp_p.setType(args.ksp_type_p) 
    ksp_p.getPC().setType(args.pc_type_p) 

    return ksp

def setup_solver2(comm, args, A):
    
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A, A)
    
    
    # ksp.setOperators(A_ass) #, A_ass)
    ksp.setType("gmres")
    ksp.setTolerances(rtol=1e-9)
    ksp.getPC().setFactorSolverType(PETSc.Mat.SolverType.MUMPS)
    # """
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)

    

    # Define the matrix blocks in the preconditioner with the velocity
    # and pressure matrix index sets

    # Return the index sets representing the row and column spaces. 2 times Blocks spaces
    nested_IS = A.getNestISs()
    ksp.getPC().setFieldSplitIS(("d", nested_IS[0][0]), ("q", nested_IS[0][1]))

    # Set the preconditioners for each block. For the top-left
    # Laplace-type operator we use algebraic multigrid. For the
    # lower-right block we use a Jacobi preconditioner. By default, GAMG
    # will infer the correct near-nullspace from the matrix block size.
    ksp_d, ksp_q = ksp.getPC().getFieldSplitSubKSP()
    ksp_d.setType(args.ksp_type_d) 
    ksp_d.getPC().setType(args.pc_type_d) 
    ksp_q.setType(args.ksp_type_q) 
    ksp_q.getPC().setType(args.pc_type_q) 

    return ksp

def compute_fp_metrics_kin(comm, args, vl, vl_):
    e_v = vl - vl_

    val_v       = assemble_scalar(form(   inner(vl, vl) *dx                 ))
    fp_err_v    = assemble_scalar(form(   inner(e_v, e_v) *dx               ))   
    
    val_v       = np.sqrt(comm.allreduce(val_v, op=MPI.SUM))
    fp_err_v    = np.sqrt(comm.allreduce(fp_err_v, op=MPI.SUM))


    return fp_err_v, val_v

def compute_fp_metrics_ela(comm, args, dl, dl_):
    e_d = dl - dl_
    K_max = np.max([args.K1, args.K2, args.K3, args.K4, args.K5]) if args.dim == 3 else np.max([args.K1, args.K2])

    val_d1  = assemble_scalar(form(   inner(grad(dl), grad(dl))*dx   ))
    val_d2  = assemble_scalar(form(   inner( div(dl), div(dl))*dx     ))
    if args.dim == 3:
        val_d3  = assemble_scalar(form(   0.5 * inner( curl(dl), curl(dl))*dx                             ))
        val_d4  = assemble_scalar(form(   0.5 *  inner( dl, curl(dl)) *  inner(dl, curl(dl))*dx             ))
        val_d5  = assemble_scalar(form(   0.5 *  inner( cross( dl, curl(dl)) , cross(dl, curl(dl))  )*dx    ))

    fp_err_d1  = assemble_scalar(form(   inner(grad(e_d), grad(e_d))*dx   ))
    fp_err_d2  = assemble_scalar(form(   inner( div(e_d), div(e_d))*dx     ))
    if args.dim == 3:
        fp_err_d3  = assemble_scalar(form(   0.5 * inner( curl(e_d), curl(e_d))*dx                             ))
        fp_err_d4  = assemble_scalar(form(   0.5 *  inner( e_d, curl(e_d)) *  inner(e_d, curl(e_d))*dx             ))
        fp_err_d5  = assemble_scalar(form(   0.5 *  inner( cross( e_d, curl(e_d)) , cross(e_d, curl(e_d))  )*dx    ))
    
    val_d1      = comm.allreduce(val_d1, op=MPI.SUM)
    val_d2      = comm.allreduce(val_d2, op=MPI.SUM)
    if args.dim == 3:
        val_d3      = comm.allreduce(val_d3, op=MPI.SUM)
        val_d4      = comm.allreduce(val_d4, op=MPI.SUM)
        val_d5      = comm.allreduce(val_d5, op=MPI.SUM)
    
    fp_err_d1      = comm.allreduce(fp_err_d1, op=MPI.SUM)
    fp_err_d2      = comm.allreduce(fp_err_d2, op=MPI.SUM)
    if args.dim == 3:
        fp_err_d3      = comm.allreduce(fp_err_d3, op=MPI.SUM)
        fp_err_d4      = comm.allreduce(fp_err_d4, op=MPI.SUM)
        fp_err_d5      = comm.allreduce(fp_err_d5, op=MPI.SUM)
    
    if args.dim == 3:
        fp_err_d   = np.sqrt( args.K1 * fp_err_d1 + args.K2 * fp_err_d2  + args.K3 * fp_err_d3  + args.K4 * fp_err_d4  + args.K5 * fp_err_d5)/np.sqrt(K_max) 
        val_d   = np.sqrt( args.K1 * val_d1 + args.K2 * val_d2  + args.K3 * val_d3  + args.K4 * val_d4  + args.K5 * val_d5)/np.sqrt(K_max) 
    else:
        fp_err_d   = np.sqrt( args.K1 * fp_err_d1 + args.K2 * fp_err_d2  )/np.sqrt(K_max) 
        val_d   = np.sqrt( args.K1 * val_d1 + args.K2 * val_d2  )/np.sqrt(K_max) 

    return  fp_err_d, val_d

def compute_metrics(comm, args, v,d, H=None, id =""):
    # ENERGY TERMS
    E_kin   = assemble_scalar(form(   0.5*inner(v, v) *dx                 ))
    E_ela1  = assemble_scalar(form(   0.5 *  inner(grad(d), grad(d))*dx   ))
    E_ela2  = assemble_scalar(form(   0.5 * inner( div(d), div(d))*dx     ))
    if args.dim == 3:
        E_ela3  = assemble_scalar(form(   0.5 * inner( curl(d), curl(d))*dx                             ))
        E_ela4  = assemble_scalar(form(   0.5 *  inner( d, curl(d)) *  inner(d, curl(d))*dx             ))
        E_ela5  = assemble_scalar(form(   0.5 *  inner( cross( d, curl(d)) , cross(d, curl(d))  )*dx    ))

    if H is not None:
        if args.chi_vert != 0.0:
            E_H_vert = assemble_scalar(form(   0.5 * inner( d, H)*inner( d, H)*dx                         ))
        if args.chi_perp != 0.0:
            E_H_perp = assemble_scalar(form(   0.5 * inner(  H, a_times_b_times_c(d, d, H) )*dx           ))

    E_kin       = comm.allreduce(E_kin, op=MPI.SUM)
    E_ela1      = comm.allreduce(E_ela1, op=MPI.SUM)
    E_ela2      = comm.allreduce(E_ela2, op=MPI.SUM)
    if args.dim == 3:
        E_ela3      = comm.allreduce(E_ela3, op=MPI.SUM)
        E_ela4      = comm.allreduce(E_ela4, op=MPI.SUM)
        E_ela5      = comm.allreduce(E_ela5, op=MPI.SUM)
    
    if H is not None:
        if args.chi_vert != 0.0:
            E_H_vert = comm.allreduce(E_H_vert, op=MPI.SUM)
        if args.chi_perp != 0.0:
            E_H_perp = comm.allreduce(E_H_perp, op=MPI.SUM)
    
    if args.dim == 3:
        E_ela   = args.K1 * E_ela1 +  args.K2 * E_ela2 + args.K3 * E_ela3 + args.K4 * E_ela4 + args.K5 * E_ela5
        E_total = E_kin + E_ela
    else:
        E_ela   = args.K1 * E_ela1 +  args.K2 * E_ela2 
        E_total = E_kin + E_ela
    

    # NODAL UNIT-NORM AND ORTHOGONALITY
    d.x.scatter_forward()
    unit1 = np.max(np.linalg.norm(np.reshape( d.x.array[:] , (-1, args.dim)), axis=1))    
    unit2 = np.min(np.linalg.norm(np.reshape( d.x.array[:] , (-1, args.dim)), axis=1))
    unit1 = comm.allreduce(unit1, op=MPI.MAX)
    unit2 = comm.allreduce(unit2, op=MPI.MIN)


    res =  {
        "Etot"+id  : E_total,
        "Ekin"+id  : E_kin,
        "Eela"+id  : E_ela,
        "Eela1"+id : E_ela1,
        "Eela2"+id : E_ela2,
        "unit1"+id  : unit1,
        "unit2"+id  : unit2
        }

    if args.dim ==3:
        res = res | { "Eela3"+id : E_ela3, "Eela4"+id : E_ela4, "Eela5"+id : E_ela5 }
    if H is not None:
        if args.chi_vert != 0.0:
            res = res | {"E_H_vert" :E_H_vert}
        if args.chi_perp != 0.0:
            res = res | {"E_H_perp" :E_H_perp}
    
    return res




