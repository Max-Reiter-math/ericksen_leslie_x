from ufl import grad, div, curl, inner, dot, cross, dx, FacetNormal, TrialFunction, TestFunction
import dolfinx.la as la
from dolfinx.fem import Function, functionspace, form, assemble_scalar, ElementMetaData, bcs_by_block, extract_function_spaces, bcs_by_block
from dolfinx.fem.petsc import assemble_matrix, assemble_matrix_nest, assemble_vector_nest, apply_lifting_nest, set_bc_nest, create_vector_nest, set_bc_nest, LinearProblem
from sim.common.operators import *
from sim.common.common_fem_methods import *
from sim.common.meta_bcs import *

def bfp_fp(comm, experiment, args, postprocess=None):
    """
    Algorithm 4.1 by Becker, R., Feng, X., & Prohl, A. (2008). Finite element approximations of the Ericksen-Leslie model for nematic liquid crystal flow. SIAM Journal on Numerical Analysis, 46(4), 1704-1731. doi:10.1137/07068254X

    Implemented using an iterative Picard-type linearization.

    Important properties of the algorithm:
    - fulfills discrete energy law
    - existence of solutions has been proven under a mesh constraint
    - asymptotic fulfillment of the unit-norm constraint under time-step restriction, see Thm. 4.4
    """

    if postprocess and comm.rank == 0:
        postprocess.log("dict", "static", {"# MPI Ranks": comm.size})

    # SECTION PARAMETERS
    dim = args.dim
    dt  = args.dt
    t   = 0

    
    if postprocess and comm.rank == 0:
        postprocess.log("dict", "static",{"SID": args.sim_id, "MODEL":args.mod,  "EXPERIMENT":args.exp, "mesh res.": args.dh, "dt":args.dt, "dim":args.dim, "T":args.T, "fp_a_tol" : args.fp_a_tol, "fp_r_tol" : args.fp_r_tol, "fp_max_iters" : args.fp_max_iters})

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
    D, Y        = functionspace(domain, ElementMetaData("Lagrange", 1 , shape=(dim,))), functionspace(domain, ElementMetaData("Lagrange", 1 , shape=(dim,)))
    TensorF     = functionspace(domain, ElementMetaData("Lagrange", 1 , shape=(dim, dim)))
    
    v1,p1, d1, q1                       = TrialFunction(P2),TrialFunction(P1), TrialFunction(D), TrialFunction(Y)
    v0,p0, d0, q0                       = Function(P2),Function(P1), Function(D), Function(Y) 
    vl,pl, dl, ql                       = Function(P2),Function(P1), Function(D), Function(Y)
    vl_, dl_                            = Function(P2), Function(D) # for inner fp iteration
    a_test, h_test, c_test, b_test      = TestFunction(P2),TestFunction(P1), TestFunction(D), TestFunction(Y)   
    
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

    # COMPUTATION OF INITIAL DIVERGENCE (BECAUSE WE USE IT TO SET THE BOUNDARY CONDITIONS FOR q)
    problem0 = LinearProblem(form(inner(qc1, bc_test)*dx), form(q_elastic_energy(args, d0, d0, d0, bc_test, H = None)),  bcs=[], u=q_ic, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
    problem0.solve()
    q0.interpolate(q_ic)
    q0.x.scatter_forward()
    
    #!SECTION

    #SECTION - BOUNDARY CONDITIONS
    bcs = []

    p_dirichlet_bcs_exist = False # NOTE - this becomes relevant later for the assembly and solver setup to decide whether a nullspace needs to be prescribed
    
    for bcwofs in boundary_conditions:
        if bcwofs.type == "Dirichlet":
            if bcwofs.quantity == "v":
                bcwofs.set_fs(P2)
                bcs.append(bcwofs.bc)
            elif bcwofs.quantity == "p":
                bcwofs.set_fs(P1)
                bcs.append(bcwofs.bc)
                p_dirichlet_bcs_exist = True
            elif bcwofs.quantity == "d":
                bcwofs.set_fs(D)
                bcs.append(bcwofs.bc)
                # BOUNDARY CONDITIONS FOR AUXILIARY VARIABLE
                #NOTE - We assume that the initial condition for the director field fulfills the boundary conditions imposed
                bcq = meta_dirichletbc("q", bcwofs.find_dofs, q_ic, marker = bcwofs.marker , entity_dim = bcwofs.dim, entities = bcwofs.entities)
                bcq.set_fs( Y)
                bcs.append(bcq.bc)
        # elif bcwofs.type == "Neumann":            
        # elif bcwofs.type == "Robin":
        else: postprocess.log("dict", "static",{"Warning" : "Boundary conditions of type "+bcwofs.type+" are currently not implemented and will be ignored..."} )
    
    #!SECTION

    #SECTION - VARIATIONAL FORM
    a, L = variational_form(dx, v1, p1, d1, q1, v0, p0, d0, q0, a_test, h_test, b_test, c_test, dl_,  dt, n_F, args, H=H, f= None,  p_dirichlet_bcs_exist = p_dirichlet_bcs_exist)
    #!SECTION
    
    #SECTION - POSTPROCESSING FOR t=0

    if postprocess:
        postprocess.log_functions(0.0, {"v": v0, "p":p0, "d":d0, "q":q0}, mesh = domain) #, meshtags = meshtags)

    metrics = compute_metrics(comm, args, v0,p0,d0,q0,  d0, d0, dx , H=H) # for initial condition
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

            # SECTION - ASSEMBLY
            measure_assembly_start = mpi_time(comm)

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
            if not p_dirichlet_bcs_exist:
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

            assembly_time += mpi_time(comm, start= measure_assembly_start)

            #!SECTION 
            
            #SECTION - SOLVER
            # SETUP SOLVERS
            start_solsetup = mpi_time(comm)
            ksp = setup_split_solver(comm, args, A, p1, h_test)
            time_solsetup += mpi_time(comm, start = start_solsetup)

            # Create finite element {py:class}`Function <dolfinx.fem.Function>`s.
            # The vectors are combined to form a  nested vector and the system is solved.
            x = PETSc.Vec().createNest([la.create_petsc_vector_wrap(vl.x), la.create_petsc_vector_wrap(pl.x), la.create_petsc_vector_wrap(dl.x), la.create_petsc_vector_wrap(ql.x)])

            # SOLVING
            start_sol = mpi_time(comm)
            ksp.solve(b, x)
            time_sol += mpi_time(comm, start = start_sol)

            #!SECTION

            #SECTION - METRICS AND UPDATING FP CRITERIA

            fp_err_v, fp_err_d, val_v, val_d = compute_fp_metrics(comm, args, vl, dl, vl_, dl_)    
            
            metrics = compute_metrics(comm, args, vl,pl,dl,ql, d0, 0.5*(dl+d0), dx, H=H)
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
            update_and_scatter([vl_, dl_], [vl, dl])
            #!SECTION

    
        #SECTION - UPDATE
        update_and_scatter([v0,p0,d0,q0], [vl,pl,dl,ql])
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
            postprocess.log_functions(t, {"v": v0, "p":p0, "d":d0, "q":q0}, mesh = domain, meshtags = meshtags)

        #!SECTION

    #!SECTION TIME EVOLUTION

    if postprocess: 
        postprocess.close()

#!SECTION GENERAL METHOD



def variational_form(dx, v1, p1, d1, q1, v0, p0, d0, q0, a, h, b, c, dl_, dt, normal_F, args, H=None, f= None, p_dirichlet_bcs_exist = False):
        """
        The variational form is structured as followed:
        a(.,.) : bilinear form describing the lhs of the system
        L(.) : linear form describing the rhs of the system
        a_ij(.,.) : bilinear form describing the lhs of the system depending on the i-th test function and the j-th trial function
        e.g.
        a_11 --> depends on v1 and a
        a_14 --> depends on q1 and a
        """

        d_ = 0.5* (dl_ + d0) # d^{j-1/2,l-1}

        # SECTION MOMENTUM EQUATION
        a_11 = inner( v1 , a )*dx # discrete time derivative

        a_11 += dt* ( inner(dot(v0, nabla_grad(v1)), a) + 0.5*div(v0)*inner(v1, a)  )*dx # temam's convection term

        if args.mu4 != 0.0:
            if args.sym_grad:
                a_11 += dt * args.mu4 *inner( grad_sym(v1), grad_sym(a))*dx
            else:            
                a_11 += dt* args.mu4 *inner( grad(v1), grad(a))*dx

        coeff1 = args.mu1 + (args.lam)**2 /args.gamma
        if coeff1 != 0.0:
            a_11 += dt * coeff1*inner(d0,dot(grad_sym(v1),d0))*inner(d0,dot(grad_sym(a),d0))*dx

        coeff2 = args.mu5 + args.mu6 - (args.lam)**2 /args.gamma
        if coeff2 != 0.0:
            a_11 += dt * coeff2 *inner( dot(grad_sym(v1),d0), dot(grad_sym(a),d0))*dx
        
        # Reformulated pressure term
        a_12 = dt * (-1)*inner(p1, div(a)) * dx 
        if p_dirichlet_bcs_exist:
            a_12 += dt * inner(p1, dot(a, normal_F)) * ds
            # NOTE - this is not the most elegant implementation
            # On the parts of the boundary where we strongly enforce the dirichlet bcs for the pressure, this will simply turn into a prescribed forcing term during assembly.
            # On the parts where this is not the case and further no-slip bcs for the velocity are employed, this term will simply vanish during assembly. 
            # This could be made more efficient by case distinction etc.

        # Ericksen stress tensor
        a_14 = (-1)*dt*inner(dot(grad(d0) , a),  q1)*dx

        # Leslie stress tensor
        if args.beta != 0.0:
            a_14 += dt* args.beta *inner(dot(grad_skw(a),d0), q1)*dx
        if args.lam != 0.0:
            a_14 -= dt* args.lam *inner(dot(grad_sym(a),d0), q1)*dx

        L_1 = inner(v0, a )*dx

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

        # Ericksen stress tensor
        a_31 = dt*inner(dot(grad(d0) , v1), c)*dx

        # Leslie stress tensor
        if args.beta != 0.0:
            a_31 -= dt* args.beta *inner(dot(grad_skw(v1),d0), c)*dx
        if args.lam != 0.0:
            a_31 += dt* args.lam *inner(dot(grad_sym(v1),d0), c)*dx


        if args.gamma != 0.0:
            a_34 = (-1)*dt* args.gamma *inner(q1, a_times_b_times_c(d_,d_,c))*dx

        L_3 = inner(d0, c)*dx

        #!SECTION DIRECTOR EQUATION

        # SECTION EQUATION FOR THE VARIATIONAL DERIVATIVE
        # TODO - recheck correct formulation here for full oseen-Frank energy
        a_43 = 0.5 * q_elastic_energy(args, d1, dl_, d0, b, H = H) 

        a_44 = (-1)* inner(q1, b)*dx 

        zero_b = Function(q0.function_space)
        L_4 = (-0.5) * q_elastic_energy(args, d0, d0, d0, b, H = H) 

        #!SECTION EQUATION FOR THE VARIATIONAL DERIVATIVE
        
        """
        Full bilinear form (lhs) and linear form (rhs)
        """
        zero_block_p =  zero_h * p1 * h * dx if p_dirichlet_bcs_exist else None # NOTE - the zero block is assembled in the case of Dirichlet boundary conditions for the pressure
        a = form([
            [a_11, a_12, None, a_14],
            [(-1)*inner(div(v1), h) * dx, zero_block_p, None, None],
            [a_31, None, a_33, a_34],
            [None, None, a_43, a_44]
        ])

        L = form([
            L_1 ,
            L_2 ,
            L_3 ,
            L_4 ,
        ])  # type: ignore

        return a, L

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

def setup_split_solver(comm, args, A, p1, h):
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
    ksp.getPC().setFieldSplitIS(("u", nested_IS[0][0]), ("p", nested_IS[0][1]), ("d", nested_IS[0][2]), ("q", nested_IS[0][3]))

    # Set the preconditioners for each block. For the top-left
    # Laplace-type operator we use algebraic multigrid. For the
    # lower-right block we use a Jacobi preconditioner. By default, GAMG
    # will infer the correct near-nullspace from the matrix block size.
    ksp_u, ksp_p, ksp_d, ksp_q = ksp.getPC().getFieldSplitSubKSP()
    ksp_u.setType(args.ksp_type_u) 
    ksp_u.getPC().setType(args.pc_type_u) 
    ksp_p.setType(args.ksp_type_p) 
    ksp_p.getPC().setType(args.pc_type_p) 
    ksp_d.setType(args.ksp_type_d) 
    ksp_d.getPC().setType(args.pc_type_d) 
    ksp_q.setType(args.ksp_type_q) 
    ksp_q.getPC().setType(args.pc_type_q) 

    return ksp

def compute_fp_metrics(comm, args, vl, dl, vl_, dl_):
    e_v = vl - vl_
    e_d = dl - dl_
    K_max = np.max([args.K1, args.K2, args.K3, args.K4, args.K5]) if args.dim == 3 else np.max([args.K1, args.K2])

    val_v       = assemble_scalar(form(   inner(vl, vl) *dx                 ))
    fp_err_v    = assemble_scalar(form(   inner(e_v, e_v) *dx               ))

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
    
    val_v       = np.sqrt(comm.allreduce(val_v, op=MPI.SUM))
    fp_err_v    = np.sqrt(comm.allreduce(fp_err_v, op=MPI.SUM))

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

    return fp_err_v, fp_err_d, val_v, val_d

def compute_metrics(comm, args, v,p,d,q, d0, d_, dx, H=None, id =""):
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
    
   

    # DISSIPATION
    dt = args.dt
    dissipation_form = dt * (args.mu1 + (args.lam)**2 /args.gamma)*(inner(d0,dot(grad_sym(v),d0))*inner(d0,dot(grad_sym(v),d0))*dx )
    dissipation_form += dt * (args.mu5 + args.mu6 - (args.lam)**2 /args.gamma) * (inner( dot(grad_sym(v),d0), dot(grad_sym(v),d0))*dx)
    if args.sym_grad:
        dissipation_form += dt * args.mu4 *inner( grad_sym(v), grad_sym(v))*dx
    else:            
        dissipation_form += dt* args.mu4 *inner( grad(v), grad(v))*dx
    dissipation_form +=(-1)*dt* args.gamma *inner(q, a_times_b_times_c(d_,d_,q))*dx

    dissipation = assemble_scalar(form( dissipation_form ))
    dissipation = comm.allreduce(dissipation, op=MPI.SUM)

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
        "diss"+id  : dissipation,
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




