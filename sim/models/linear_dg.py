from ufl import grad, div, inner, dot, avg, jump, dx, FacetNormal, CellDiameter, Measure, TrialFunction, TestFunction
import dolfinx.la as la
from dolfinx.fem import Function, functionspace, form, assemble_scalar, ElementMetaData, bcs_by_block, extract_function_spaces, bcs_by_block
from dolfinx.fem.petsc import assemble_matrix, assemble_matrix_nest, assemble_vector_nest, apply_lifting_nest, set_bc_nest, create_vector_nest, set_bc_nest, LinearProblem
from sim.common.grad_dg0 import discrete_gradient_def, d_to_grad_d_mappings, reconstruct_grad, interpolate_dg0_at, compute_dg0_int_pts_on_bdry
from sim.common.operators import *
from sim.common.common_fem_methods import *
from sim.common.meta_bcs import *

def linear_dg(comm, experiment, args, postprocess=None):
    """
    Algorithm 2 by Maximilian E. V. Reiter. (2025). Projection Methods in the Context of Nematic Crystal Flow.

    Important properties of the algorithm:
    - fulfills a discrete energy law
    - unconditional existence
    - automatically fulfills unit-norm constraint
    - assumes restrictive mesh-conditions
    """

    if postprocess and comm.rank == 0:
        postprocess.log("dict", "static", {"# MPI Ranks": comm.size})
        
    # SECTION PARAMETERS
    dim = args.dim
    dt  = args.dt
    t   = 0
    
    if postprocess and comm.rank == 0:
        postprocess.log("dict", "static",{"SID": args.sim_id, "MODEL":args.mod, "PROJECTION STEP": args.projection_step, "PROJECT TANGENT MATRIX": args.project_tangent_map, "ALPHA" : args.alpha, "EXPERIMENT":args.exp, "mesh res.": args.dh, "dt":args.dt, "dim":args.dim, "T": args.T})

    #!SECTION PARAMETERS
    domain          = experiment.mesh    
    meshtags        = experiment.meshtags
    dg0_cells       = experiment.dg0_cells
    dg0_int_points  = experiment.dg0_int_points
    dx          = Measure('dx', domain=domain) 
    ds          = Measure('ds', domain=domain, subdomain_data=meshtags) # since we allow for mix of boundary conditions                                         
    dS          = Measure('dS', domain=domain)
    n_F         = FacetNormal(domain)
    h_T         = CellDiameter(domain)

    #add all local variables for transparency    
    if postprocess and comm.rank == 0:
        postprocess.log("dict","static",{"model.vars":dict(locals())}, visible =False)

    initial_conditions          = experiment.initial_conditions
    boundary_conditions         = experiment.boundary_conditions

    total_time_start = mpi_time(comm)
    
    # SECTION - FUNCTION SPACES AND FUNCTIONS

    P2          = functionspace(domain, ElementMetaData("Lagrange", 2 , shape=(dim,))) 
    P1          = functionspace(domain, ("Lagrange", 1) )
    D, Y        = functionspace(domain, ElementMetaData("DG", 0 , shape=(dim,))), functionspace(domain, ElementMetaData("DG", 0 , shape=(dim,)))
    TensorF     = functionspace(domain, ElementMetaData("DG", 0 , shape=(dim, dim))) # function space for the reconstructed gradient
    
    v1,p1, d1, grad_d1, q1                  = TrialFunction(P2),TrialFunction(P1),TrialFunction(D),TrialFunction(TensorF), TrialFunction(Y)
    v0,p0, d0, grad_d0, q0                  = Function(P2),Function(P1), Function(D), Function(TensorF), Function(Y)
    d_                                      = Function(D) # d_ is used for the tangent projection
    d_bc                                    = Function(D) # d_bc is used for the weak anchoring of the dirichlet boundary condition
    v,p, d, grad_d, q                       = Function(P2),Function(P1), Function(D), Function(TensorF), Function(Y)
    a_test, h_test, c_test, tau, b_test     = TestFunction(P2),TestFunction(P1), TestFunction(D), TestFunction(TensorF), TestFunction(Y)      
    
    # FOR COMPUTATION OF INITIAL q
    qc1 = TrialFunction(Y)
    q_ic = Function(Y)
    bc_test = TestFunction(Y)  

    # COMPUTE AND SAVE DOFS PER RANK
   
    local_dofs = P2.dofmap.index_map.size_local + P1.dofmap.index_map.size_local + D.dofmap.index_map.size_local + Y.dofmap.index_map.size_local    # Count DOFs on this rank    
    local_info = {f"rank {comm.rank} dofs": local_dofs}                                                                                             # Create local dictionary   
    all_info = comm.gather(local_info, root=0)                                                                                                      # Gather all dictionaries at root

    if comm.rank == 0:
        # Merge all into a single dict
        combined = {}
        for partial_dict in all_info:
            combined.update(partial_dict)
        postprocess.log("dict", "static", combined)

    #!SECTION FUNCTION SPACES AND FUNCTIONS
    
    

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
            elif bcwofs.type == "Dirichlet" and bcwofs.quantity == "d":
                #FIXME - wrong interpolation points!
                b_cells, b_int_pts = compute_dg0_int_pts_on_bdry(domain, dg0_cells, dg0_int_points, marker = bcwofs.marker)
                interpolate_dg0_at(d_bc, bcwofs.values, b_cells, b_int_pts) # DG version of interpolation
        # elif bcwofs.type == "Neumann":            
        # elif bcwofs.type == "Robin":
        else: postprocess.log("dict", "static",{"Warning" : "Boundary conditions of type "+bcwofs.type+" are currently not implemented and will be ignored..."} )
    
    #!SECTION

    # SECTION VARIATONAL FORMULATION
    a, L = variational_form(D, TensorF, v1, p1, d1, q1, grad_d1, v0, p0, d0, q0, grad_d0, a_test, h_test, b_test, c_test, tau, d_, d_bc, dt, n_F, h_T, args, ds, dS, H=None, f= None, boundary_conditions=boundary_conditions, postprocess = postprocess, dg0_cells = dg0_cells, dg0_int_points = dg0_int_points, p_dirichlet_bcs_exist = p_dirichlet_bcs_exist)
    
    # SECTION INITIAL CONDITIONS
    v0.interpolate(initial_conditions["v"]) 
    p0.interpolate(initial_conditions["p"]) 
    interpolate_dg0_at(d0, initial_conditions["d"], dg0_cells, dg0_int_points) # DG version of interpolation
    interpolate_dg0_at(d_, initial_conditions["d"], dg0_cells, dg0_int_points) # DG version of interpolation
    scatter_all([v0, p0, d0, d_, d_bc])
    # Create Matrix and Vector for the stationary Equation System to reconstruct the gradient
    B, res = d_to_grad_d_mappings(TensorF, grad_d1, tau, D , d1, boundary_conditions, n_F, ds,dS, cells = dg0_cells, int_points = dg0_int_points )
    # Compute reconstructed gradient for t = 0
    reconstruct_grad(B,res, grad_d0, d0) # NOTE - the result is saved to the function grad_d0
    
    # COMPUTATION OF INITIAL DIVERGENCE (BECAUSE WE USE IT TO SET THE BOUNDARY CONDITIONS FOR q)
    # d0 and grad_d0 already exist, it suffices to compute q from that accordingly
    #TODO - add function
    a_45, a_43, L_4 = discrete_laplacian_def(grad_d0, d0, b_test, d_bc, n_F, args.alpha, h_T, boundary_conditions, ds, dS)
    problem0 = LinearProblem(form(inner(qc1, bc_test)*dx), form(L_4 - a_45 - a_43),  bcs=[], u=q_ic, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
    problem0.solve()
    q0.interpolate(q_ic)
    q0.x.scatter_forward()
    
    #!SECTION

    #SECTION - POSTPROCESSING FOR t=0
    DG1  = functionspace(domain, ElementMetaData("DG", 1 , shape=(dim,))) # space used to output to vtx
    DG1T = functionspace(domain, ElementMetaData("DG", 1 , shape=(dim,dim)))
    d_out, q_out, dbc_out = Function(DG1), Function(DG1), Function(DG1) # vtx cant deal with dg0, but with dg1
    grad_d_out = Function(DG1T)
    # NOTE - here the classical interpolation is sufficient since we simply save DG0 functions in a DG1 space. The interpolation is therefore exact.
    d_out.interpolate(d0)
    q_out.interpolate(q0)
    dbc_out.interpolate(d_bc)
    grad_d_out.interpolate(grad_d0)
    scatter_all([d_out, q_out, grad_d_out,dbc_out])
    if postprocess:
        postprocess.log_functions(0.0, {"v": v0, "p":p0, "d":d_out, "grad_d": grad_d_out, "q":q_out , "dbc": dbc_out}, mesh = domain) #, meshtags = meshtags)

    metrics = compute_metrics(comm, args, v0,p0,d0,grad_d0, d0,q0, d_bc, h_T, ds, dS, id ="", postprocess = postprocess) # for initial condition
    if postprocess and comm.rank == 0:
        postprocess.log("dict", t, { "time" : t} |  metrics )

    #!SECTION

    # SECTION TIME EVOLUTION
    total_time = mpi_time(comm, start = total_time_start )

    while t < args.T:
        t += dt 

        # SECTION - ASSEMBLY
        measure_assembly_start = mpi_time(comm)
        A, b = assemble_all(a, L, B, res, bcs = bcs, p_dirichlet_bcs_exist = p_dirichlet_bcs_exist)
        assembly_time = mpi_time(comm, start= measure_assembly_start)
        #!SECTION 
        
        #SECTION - SOLVER
        # SETUP SOLVERS
        start_solsetup = mpi_time(comm)
        ksp = setup_split_solver(comm, args, A, p1, h_test)
        time_solsetup = mpi_time(comm, start = start_solsetup)

        # The vectors are combined to form a  nested vector and the system is solved.
        x = PETSc.Vec().createNest([la.create_petsc_vector_wrap(v.x), la.create_petsc_vector_wrap(p.x), la.create_petsc_vector_wrap(d.x), la.create_petsc_vector_wrap(q.x)])

        # SOLVING
        start_sol = mpi_time(comm)
        
        ksp.solve(b, x)
        d.x.scatter_forward()
        # Compute reconstructed DG= gradient afterwards
        reconstruct_grad(B,res, grad_d, d)
        
        time_sol = mpi_time(comm, start = start_sol)

        #!SECTION

        #SECTION - EVTL. METRICS BEFORE PROJECTION STEP    
        if args.projection_step ==1:
            metrics = compute_metrics(comm, args, v,p,d,grad_d, d0,q, d_bc, h_T, ds, dS, id=".b4p", postprocess = postprocess)
            if postprocess and comm.rank == 0:
                postprocess.log("dict", t, { "time" : t} | metrics , visible = False)
                postprocess.log("dict", t, {"time":t, "t.ass": assembly_time, "t.sol": time_sol, "t.solsetup" : time_solsetup}, visible = True)

        #!SECTION

        # SECTION - NODAL PROJECTION STEP
        if args.projection_step ==1:
            start_pstep = mpi_time(comm)

            nodal_normalization(d, dim)

            time_pstep = mpi_time(comm, start = start_pstep )
        #!SECTION 

        #SECTION - UPDATE
        update_and_scatter([v0,p0,d0,d_,q0, grad_d0], [v,p,d,d,q,grad_d])
        reconstruct_grad(B,res, grad_d0, d0)
        #!SECTION 

        # SECTION - NODAL PROJECTION STEP FOR TANGENT MAP
        if args.project_tangent_map == 1:
            start_pstep2 = mpi_time(comm)

            nodal_normalization(d, dim)

            time_pstep2 = mpi_time(comm, start = start_pstep2 )
        #!SECTION 

        
        
        #SECTION - METRICS AT END OF ITERATION 
        metrics =  compute_metrics(comm, args, v,p,d,grad_d, d0,q, d_bc, h_T, ds, dS, postprocess = postprocess)

        errorL2 = np.nan
        if experiment.has_exact_solution:   
            errorL2 = experiment.compute_error(comm, d,t,norm = "L2", degree_raise = 3, family = "DG")   

        total_time = mpi_time(comm, start = total_time_start )
        if postprocess and comm.rank == 0:
            postprocess.log("dict", t, { "time" : t, "errorL2" : errorL2 , "t.tot" : total_time} | metrics)
            if args.projection_step ==1: postprocess.log("dict", t, { "time" : t, "t.pstep" : time_pstep})
            if args.project_tangent_map == 1: postprocess.log("dict", t, { "time" : t, "t.pstep2" : time_pstep2})
        
        #!SECTION

        #SECTION - SAVING
        d_out.interpolate(d0)
        q_out.interpolate(q0)
        dbc_out.interpolate(d_bc)
        grad_d_out.interpolate(grad_d0)
        scatter_all([d_out, q_out, grad_d_out,dbc_out])
        if postprocess: 
            postprocess.log_functions(t, {"v": v0, "p":p0, "d":d_out, "grad_d": grad_d_out, "q":q_out, "dbc": dbc_out}, mesh = domain, meshtags = meshtags)

        #!SECTION

    #!SECTION TIME EVOLUTION

    if postprocess: 
        postprocess.close()

#!SECTION GENERAL METHOD



def variational_form(DFS, TFS, v1, p1, d1, q1, grad_d1, v0, p0, d0, q0, grad_d0, a, h, b, c, tau, d_, d_bc, dt, normal_F, h_T, args, ds, dS, H=None, f= None, boundary_conditions=[], postprocess = None, dg0_cells: np.ndarray = None, dg0_int_points: np.ndarray = None, p_dirichlet_bcs_exist = False):
        """
        The variational form is structured as followed:
        a(.,.) : bilinear form describing the lhs of the system
        L(.) : linear form describing the rhs of the system
        a_ij(.,.) : bilinear form describing the lhs of the system depending on the i-th test function and the j-th trial function
        e.g.
        a_11 --> depends on v1 and a
        a_14 --> depends on q1 and a
        a_54 --> depends on grad_d1 and b
        """

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
            a_11 += dt * coeff1*inner(d_,dot(grad_sym(v1),d_))*inner(d_,dot(grad_sym(a),d_))*dx

        coeff2 = args.mu5 + args.mu6 - (args.lam)**2 /args.gamma
        if coeff2 != 0.0:
            a_11 += dt * coeff2 *inner( dot(grad_sym(v1),d_), dot(grad_sym(a),d_))*dx

        # Reformulated pressure term
        a_12 = (-1)*inner(p1, div(a)) * dx 
        if p_dirichlet_bcs_exist:
            a_12 += inner(p1, dot(a, normal_F)) * ds
            # NOTE - this is not the most elegant implementation
            # On the parts of the boundary where we strongly enforce the dirichlet bcs for the pressure, this will simply turn into a prescribed forcing term during assembly.
            # On the parts where this is not the case and further no-slip bcs for the velocity are employed, this term will simply vanish during assembly. 
            # This could be made more efficient by case distinction etc.

        # Ericksen stress tensor
        a_14 = dt*inner(dot(grad_d0 , a), a_times_b_times_c(d_,d_, q1))*dx

        # Leslie stress tensor
        if args.beta != 0.0:
            a_14 -= dt* args.beta *inner(dot(grad_skw(a),d_), a_times_b_times_c(d_,d_,q1))*dx
        if args.lam != 0.0:
            a_14 += dt* args.lam *inner(dot(grad_sym(a),d_), a_times_b_times_c(d_,d_,q1))*dx

        L_1 = inner(v0, a )*dx

        if f is not None:
            L_1 += dt*inner(f, a)*dx

        #!SECTION MOMENTUM EQUATION

        # SECTION DIVERGENCE ZERO CONDITION
        # Can be found in the final variational formulation
        
        zero_h = Function(p0.function_space)
        L_2 = inner(zero_h, h) * dx

        #!SECTION DIVERGENCE ZERO CONDITION

        # SECTION DIRECTOR EQUATION
        a_33 = inner(d1, c)*dx

        # Ericksen stress tensor
        a_31 = (-1)*dt*inner(dot(grad_d0 , v1), a_times_b_times_c(d_,d_, c))*dx

        # Leslie stress tensor
        if args.beta != 0.0:
            a_31 += dt* args.beta *inner(dot(grad_skw(v1),d_), a_times_b_times_c(d_,d_,c))*dx
        if args.lam != 0.0:
            a_31 -= dt* args.lam *inner(dot(grad_sym(v1),d_), a_times_b_times_c(d_,d_,c))*dx

        if args.gamma != 0.0:
            a_34 = (-1)*dt* args.gamma *inner(q1, a_times_b_times_c(d_,d_,c))*dx

        L_3 = inner(d0, c)*dx

        #!SECTION DIRECTOR EQUATION

        # SECTION EQUATION FOR THE VARIATIONAL DERIVATIVE
        # mass matrix for q
        a_44 = (-1)* inner(q1, b)*dx 

        a_45, a_43, L_4 = discrete_laplacian_def(grad_d1, d1, b, d_bc, normal_F, args.alpha, h_T, boundary_conditions, ds, dS)
        
        # evtl. add energy parts that depend on the magnetic field
        if H != None and args.chi_vert != 0.0:
            a_43 -= args.chi_vert * inner( d1, H)*inner( b, H)*dx
        if H != None and args.chi_perp != 0.0:
            a_43 += args.chi_perp * inner(  H, a_times_b_times_c(d1, b, H) )*dx

        """
        RECONSTRUCTED GRADIENT (All terms dependent on test function tau)
        """
        
        a_55 = (-1)*args.K1 * inner( grad_d1, tau)*dx

        if args.K2 != 0.0 or args.K3 != 0.0 or args.K4 != 0.0 or args.K5 != 0.0:
            if postprocess:
                postprocess.log_message("Values other than 0 for K2,K3,K4,K5 are not supported for this model")
        
        # NOTE - the following forms are only filled to better understand the full system. At this point they are not necessary for the variational formulation.
        a_53, L_5 = None, None
        # a_53, L_5 = discrete_gradient_def(grad_d_FS = TFS, tau_test= tau, d_FS = DFS, d_trial = d1, boundary_conditions=boundary_conditions, normal_F=normal_F,  ds=ds, dS=dS, cells = dg0_cells, int_points = dg0_int_points)


        #!SECTION EQUATION FOR THE VARIATIONAL DERIVATIVE
        
        """
        Full bilinear form (lhs) and linear form (rhs)
        """
        zero_block_p =  zero_h * p1 * h * dx if p_dirichlet_bcs_exist else None # NOTE - the zero block is assembled in the case of Dirichlet boundary conditions for the pressure
        a = [
            [a_11, a_12, None, a_14, None],
            [(-1)*inner(div(v1), h) * dx, zero_block_p, None, None, None],
            [a_31, None, a_33, a_34, None],
            [None, None, a_43, a_44, a_45],
            [None, None, a_53, None, a_55],
        ]

        L = [
            L_1 ,
            L_2 ,
            L_3 ,
            L_4 ,
            L_5 ,
        ] 

        return a, L



def discrete_laplacian_def(grad_d_trial, d_trial, b_test, d_bc, normal_F, alpha, h_T, boundary_conditions, ds, dS ):
    """
    Variational Formulation of the RHS of the discrete Laplacian, i.e.
        inner(q,b_test)*dxL
        =
        a_45 + a_43 - L4
    """
    # NOTE - ds needs to be given as argument due to the meshtags being initialized prior
    # NOTE - no parallel architecture needed here, since we only initialize a variational formulation

    # applying the Definition of the discrete lifting onto the discrete gradient of the test function b
    # NOTE - using the normal in direction '-' is consistent with the Definition of the discrete gradient and is necessary for the right results!
    a_45 = inner(dot(avg(grad_d_trial),normal_F('-')),jump(b_test))*dS      # on the interior

    # Penalization terms
    a_43 = (alpha/avg(h_T)) * inner( jump(d_trial), jump(b_test) )*dS   # interior jump penalization 

    # initialize the rhs in this row with zero
    zero_vec = Function(d_bc.function_space)
    L_4 = inner(zero_vec, b_test)*dx

    # boundary condition
    for bcwofs in boundary_conditions:
        if bcwofs.type == "Dirichlet" and bcwofs.quantity == "d":
            a_45 += -inner(dot(grad_d_trial, normal_F), b_test)*ds(bcwofs.meshtag)             # Definition of the discrete lifting on the boundary        
            a_43 += (alpha/h_T)*inner(d_trial, b_test)*ds(bcwofs.meshtag)     # bc penalization
            L_4 += (alpha/h_T)*inner( d_bc, b_test)*ds(bcwofs.meshtag)     # bc penalization
    
    return a_45, a_43, L_4


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
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE) # NOTE - this time we use additive since it is better in parallel

    # Return the index sets representing the row and column spaces. 2 times Blocks spaces
    nested_IS = P.getNestISs()
    ksp.getPC().setFieldSplitIS(("u", nested_IS[0][0]), ("p", nested_IS[0][1]), ("d", nested_IS[0][2]), ("q", nested_IS[0][3]))

    # Set the preconditioners for each block via CLI.
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

def assemble_all(a, L, B, res, bcs = [], p_dirichlet_bcs_exist = False):
    """
    Recall that
    a = [   [a_11, inner(p1, div(a)) * dx, None, a_14, None],
            [inner(div(v1), h) * dx, None, None, None, None],
            [a_31, None, a_33, a_34, None],
            [None, None, a_43, a_44, a_45],
            [None, None, a_53, None, a_55] ]
    L = [L_1 , L_2 , L_3 , L_4 , L_5 ] 
    """
    a_reduced = form([[a[i][j] for j in range(4)] for i in range(4)])
    L_reduced = form([L[i] for i in range(4)])
    # Assemble nested matrix operators
    A = assemble_matrix_nest(a_reduced, bcs=bcs)
    A.assemble()        
    b = assemble_vector_nest(L_reduced) 
    apply_lifting_nest(b, a_reduced, bcs=bcs)

    # Ghost update
    b_sub_vecs = b.getNestSubVecs()
    for b_sub in b_sub_vecs:
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Set Dirichlet boundary condition values in the RHS vector
    bcs0 = bcs_by_block(extract_function_spaces(L_reduced), bcs)
    set_bc_nest(b, bcs0)

    
    a_45 = form(a[3][4])
    """
    The last row can be ignored, since we alredy initialized the map from d.x to grad_d.x in the beginning with the method: 
        B, res = d_to_grad_d_mappings(TensorF, grad_d1, tau, d1, d_bc, n_F, boundary_conditions = [])

    Accordingly, we do not invoke:
        a_53 = form(a[4][2])
        a_55 = form(a[4][4])
    """   
    A_45 = assemble_matrix(a_45, bcs=bcs)
    A_45.assemble()
    """
    Recall, how matrix B and vector res map from d.x to grad_d.x:
    grad_d.x = B d.x + res
    """
    A_45res = b.getNestSubVecs()[3].copy() # get nested vector
    res.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    A_45.mult(res, A_45res)  # multiply A_45res = A45 res
    b_sub_vecs[3].axpy(-1.0, A_45res) # b4 <- b4 - A_45*res
    b_sub_vecs[3].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    b.assemble()
    
    A_45B = A_45.matMult(B) # A_45 <- A_45 *B

    A_43 = A.getNestSubMatrix(3, 2)
    A_43.axpy(1.0, A_45B) #, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
    A.assemble() # since matrix was manually modified, we need to reassemble
   

    # Since the variational formulation includes only the pressure gradient, the pressure is only fixed up to a constant.
    # This is only true, unless Dirichlet boundary conditions for the pressure are prescribed
    if not p_dirichlet_bcs_exist:
        # We supply a vector that spans the nullspace to the solver, and any component
        # of the solution in this direction will be eliminated during the
        # solution process.
        null_vec = create_vector_nest(L_reduced)

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

    return A,b

def compute_metrics(comm, args, v,p,d,grad_d, d0,q, dbc, h_T, ds, dS,  id ="", postprocess = None):
    # ENERGY TERMS
    E_kin   = assemble_scalar(form(   0.5 *                inner(v, v) *dx                    ))
    E_Ji    = assemble_scalar(form(   0.5 * (1/avg(h_T)) * inner( jump(d), jump(d) )*dS       ))
    E_Jbc   = assemble_scalar(form(   0.5 * (1/h_T)      * inner( d-dbc, d-dbc )*ds           ))
    E_ela1  = assemble_scalar(form(   0.5 *                inner(grad_d, grad_d)*dx           ))

    if args.K2 != 0.0 or args.K3 != 0.0 or args.K4 != 0.0 or args.K5 != 0.0:
        if postprocess:
            postprocess.log_message("Values other than 0 for K2,K3,K4,K5 are not supported for this model")
    
    
    E_kin       = comm.allreduce(E_kin, op=MPI.SUM)
    E_Ji        = comm.allreduce(E_Ji, op=MPI.SUM)
    E_Jbc       = comm.allreduce(E_Jbc, op=MPI.SUM)
    E_ela1      = comm.allreduce(E_ela1, op=MPI.SUM)
    E_ela   = args.K1 * E_ela1 + args.alpha*E_Ji + args.alpha*E_Jbc
    E_total = E_kin + E_ela
    
    #TODO - Magnetic field

    # DISSIPATION
    dt = args.dt
    dissipation_form = dt * (args.mu1 + (args.lam)**2 /args.gamma)*(inner(d0,dot(grad_sym(v),d0))*inner(d0,dot(grad_sym(v),d0))*dx )
    dissipation_form += dt * (args.mu5 + args.mu6 - (args.lam)**2 /args.gamma) * (inner( dot(grad_sym(v),d0), dot(grad_sym(v),d0))*dx)
    if args.sym_grad:
        dissipation_form += dt * args.mu4 *inner( grad_sym(v), grad_sym(v))*dx
    else:            
        dissipation_form += dt* args.mu4 *inner( grad(v), grad(v))*dx
    dissipation_form +=(-1)*dt* args.gamma *inner(q, a_times_b_times_c(d0,d0,q))*dx

    dissipation = assemble_scalar(form( dissipation_form ))
    dissipation = comm.allreduce(dissipation, op=MPI.SUM)

    # NODAL UNIT-NORM AND ORTHOGONALITY
    d.x.scatter_forward()
    d0.x.scatter_forward()
    orthogonality = np.max(np.abs( np.sum( (np.reshape( d.x.array[:] , (-1, args.dim)) - np.reshape( d0.x.array[:] , (-1, args.dim))) * np.reshape( d0.x.array[:] , (-1, args.dim)) , axis=1 ) ))
    orthogonality = comm.allreduce(orthogonality, op=MPI.MAX)
    unit1 = np.max(np.linalg.norm(np.reshape( d.x.array[:] , (-1, args.dim)), axis=1))    
    unit2 = np.min(np.linalg.norm(np.reshape( d.x.array[:] , (-1, args.dim)), axis=1))
    unit1 = comm.allreduce(unit1, op=MPI.MAX)
    unit2 = comm.allreduce(unit2, op=MPI.MIN)


    res =  {
        "Etot"+id  : E_total,
        "Ekin"+id  : E_kin,
        "Eela"+id  : E_ela,
        "EJi"+id    : E_Ji,
        "EJbc"+id  : E_Jbc,
        "Eela1"+id : E_ela1,
        "diss"+id  : dissipation,
        "orth"+id  : orthogonality,
        "unit1"+id  : unit1,
        "unit2"+id  : unit2
        }
    
    return res




