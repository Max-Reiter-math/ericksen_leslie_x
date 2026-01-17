from ufl import grad, div, curl, inner, dot, cross, dx, Measure, FacetNormal, TrialFunction, TestFunction
import dolfinx.la as la
from dolfinx.fem import Function, functionspace, form, assemble_scalar, ElementMetaData, bcs_by_block, extract_function_spaces, bcs_by_block
from dolfinx.fem.petsc import assemble_matrix, assemble_matrix_nest, assemble_vector_nest, apply_lifting_nest, set_bc_nest, create_vector_nest, set_bc_nest, LinearProblem
from sim.common.operators import *
from sim.common.common_fem_methods import *
from sim.common.meta_bcs import *

def linear_cg(comm, experiment, args, postprocess=None):
    """
    Algorithm 1 by Maximilian E. V. Reiter. (2025). Projection Methods in the Context of Nematic Crystal Flow.

    Important properties of the algorithm:
    - fulfills a discrete energy law
    - unconditional existence
    - automatically fulfills unit-norm constraint
    - assumes a weakly-acute mesh
    """

    if postprocess and comm.rank == 0:
        postprocess.log("dict", "static", {"# MPI Ranks": comm.size})

    # SECTION PARAMETERS
    dim = args.dim
    dt  = args.dt
    t   = 0
    
    if postprocess and comm.rank == 0:
        postprocess.log("dict", "static",{"SID": args.sim_id, "MODEL":args.mod, "PROJECTION STEP": args.projection_step, "PROJECT TANGENT MATRIX": args.project_tangent_map, "MASS LUMPING" : args.mass_lumping, "EXPERIMENT":args.exp, "mesh res.": args.dh, "dt":args.dt, "dim":args.dim, "T": args.T})

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
    v0,p0, d0, d_, q0                       = Function(P2),Function(P1), Function(D), Function(D), Function(Y) # d_ is used for the tangent projection
    v,p, d, q                           = Function(P2),Function(P1), Function(D), Function(Y)
    a_test, h_test, c_test, b_test      = TestFunction(P2),TestFunction(P1), TestFunction(D), TestFunction(Y)
    grad_d0_project                     = Function(TensorF)        
    
    # FOR COMPUTATION OF INITIAL q
    qc1 = TrialFunction(Y)
    q_ic = Function(Y)
    bc_test = TestFunction(Y)  

    # COMPUTE AND SAVE DOFS PER RANK
   
    local_dofs = P2.dofmap.index_map.size_local + P1.dofmap.index_map.size_local + D.dofmap.index_map.size_local + Y.dofmap.index_map.size_local    # Count DOFs on this rank    
    local_info = {f"rank {comm.rank} dofs": local_dofs}                                                                                           # Create local dictionary   
    all_info = comm.gather(local_info, root=0)                                                                                                      # Gather all dictionaries at root

    if comm.rank == 0:
        # Merge all into a single dict
        combined = {}
        for partial_dict in all_info:
            combined.update(partial_dict)
        postprocess.log("dict", "static", combined)

    #!SECTION FUNCTION SPACES AND FUNCTIONS
    
    # SECTION VARIATONAL FORMULATION
    
    if args.mass_lumping:
        dxL = Measure("dx", domain = domain, metadata = {"quadrature_rule": "vertex", "quadrature_degree": 0})
        grad_d0 = grad_d0_project
    else:
        dxL = dx
        grad_d0 = grad(d0)

    # SECTION INITIAL CONDITIONS
    v0.interpolate(initial_conditions["v"]) 
    p0.interpolate(initial_conditions["p"]) 
    d0.interpolate(initial_conditions["d"]) 
    d_.interpolate(initial_conditions["d"]) 
    if "H" in initial_conditions.keys():
        H = Function(D)
        H.interpolate(initial_conditions["H"])
        # TODO - scatter forward
    else:
        H = None
    scatter_all([v0, p0, d0, d_])

    if args.mass_lumping: 
        grad_d0_project.interpolate(project_lumped(grad(d0),TensorF))
        grad_d0_project.x.scatter_forward()
    
    # COMPUTATION OF INITIAL DIVERGENCE (BECAUSE WE USE IT TO SET THE BOUNDARY CONDITIONS FOR q)
    problem0 = LinearProblem(form(inner(qc1, bc_test)*dxL), form(q_elastic_energy(args, d0, d0, bc_test, H = None)),  bcs=[], u=q_ic, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
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
    a, L = variational_form(dxL, v1, p1, d1, q1, v0, p0, d0, q0, a_test, h_test, b_test, c_test, grad_d0, d_,  dt, n_F, args, H=H, f= None, p_dirichlet_bcs_exist = p_dirichlet_bcs_exist)
    #!SECTION
    
    #SECTION - POSTPROCESSING FOR t=0

    if postprocess:
        postprocess.log_functions(0.0, {"v": v0, "p":p0, "d":d0, "q":q0}, mesh = domain) #, meshtags = meshtags)

    metrics = compute_metrics(comm, args, v0,p0,d0,d0,q0, dxL , H = H) # for initial condition
    if postprocess and comm.rank == 0:
        postprocess.log("dict", t, { "time" : t} |  metrics )

    #!SECTION

    # SECTION TIME EVOLUTION
    total_time = mpi_time(comm, start = total_time_start )

    while t < args.T:
        t += dt 

        # SECTION - ASSEMBLY
        measure_assembly_start = mpi_time(comm)

        # Assemble nested matrix operators
        # NOTE - the following is based on the dolfinx tutorials        
        A = assemble_matrix_nest(a, bcs=bcs)
        A.assemble()

        # Assemble right-hand side vector
        b = assemble_vector_nest(L)

        # Modify ('lift') the RHS for Dirichlet boundary conditions
        apply_lifting_nest(b, a, bcs=bcs)

        # Ghost Updates
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
            null_vecs[0].set(0.0) 
            null_vecs[1].set(1.0)

            # Normalize the vector that spans the nullspace, create a nullspace
            # object, and attach it to the matrix
            null_vec.normalize()
            nsp = PETSc.NullSpace().create(vectors=[null_vec])
            assert nsp.test(A) 
            A.setNullSpace(nsp)

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
        time_sol = mpi_time(comm, start = start_sol)

        #!SECTION

        #SECTION - EVTL. METRICS BEFORE PROJECTION STEP  
        if postprocess and comm.rank == 0:
            postprocess.log("dict", t, {"time":t, "t.ass": assembly_time, "t.sol": time_sol, "t.solsetup" : time_solsetup}, visible = True)  
            
        if args.projection_step ==1 or args.project_tangent_map ==1:
            metrics = compute_metrics(comm, args, v,p,d,d0,q, dxL, H=H, id=".b4p")
            if postprocess and comm.rank == 0:
                postprocess.log("dict", t, { "time" : t} | metrics , visible = False)

        #!SECTION

        # SECTION - NODAL PROJECTION STEP
        if args.projection_step ==1:
            start_pstep = mpi_time(comm)

            nodal_normalization(d, dim)

            time_pstep = mpi_time(comm, start = start_pstep )
        #!SECTION 

        #SECTION - UPDATE
        update_and_scatter([v0,p0,d0,d_,q0], [v,p,d,d,q])
        if args.mass_lumping: 
            grad_d0_project.interpolate(project_lumped(grad(d0),TensorF))
            grad_d0_project.x.scatter_forward()
        #!SECTION 

        # SECTION - NODAL PROJECTION STEP FOR TANGENT MAP
        if args.project_tangent_map == 1:
            start_pstep2 = mpi_time(comm)

            nodal_normalization(d_, dim)

            time_pstep2 = mpi_time(comm, start = start_pstep2 )
        #!SECTION 

        
        
        #SECTION - METRICS AT END OF ITERATION 
        metrics =  compute_metrics(comm, args, v,p,d,d0,q, dxL , H=H)

        errorL2 = np.nan
        if experiment.has_exact_solution:   
            errorL2 = experiment.compute_error(comm, d,t,norm = "L2", degree_raise = 3)   

        total_time = mpi_time(comm, start = total_time_start )
        if postprocess and comm.rank == 0:
            postprocess.log("dict", t, { "time" : t, "errorL2" : errorL2 , "t.tot" : total_time} | metrics)
            if args.projection_step ==1: postprocess.log("dict", t, { "time" : t, "t.pstep" : time_pstep})
            if args.project_tangent_map == 1: postprocess.log("dict", t, { "time" : t, "t.pstep2" : time_pstep2})
        
        #!SECTION

        #SECTION - SAVING
        if postprocess: 
            postprocess.log_functions(t, {"v": v0, "p":p0, "d":d0, "q":q0}, mesh = domain, meshtags = meshtags)

        #!SECTION

    #!SECTION TIME EVOLUTION

    if postprocess: 
        postprocess.close()

#!SECTION GENERAL METHOD



def variational_form(dxL, v1, p1, d1, q1, v0, p0, d0, q0, a, h, b, c, grad_d0, d_, dt, normal_F, args, H=None, f= None, p_dirichlet_bcs_exist = False):
        """
        The variational form is structured as followed:
        a(.,.) : bilinear form describing the lhs of the system
        L(.) : linear form describing the rhs of the system
        a_ij(.,.) : bilinear form describing the lhs of the system depending on the i-th test function and the j-th trial function
        e.g.
        a_11 --> depends on v1 and a
        a_14 --> depends on q1 and a
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
        a_14 = dt*inner(dot(grad_d0 , a), a_times_b_times_c(d_,d_, q1))*dxL

        # Leslie stress tensor
        if args.beta != 0.0:
            a_14 -= dt* args.beta *inner(dot(grad_skw(a),d_), a_times_b_times_c(d_,d_,q1))*dxL
        if args.lam != 0.0:
            a_14 += dt* args.lam *inner(dot(grad_sym(a),d_), a_times_b_times_c(d_,d_,q1))*dxL

        L_1 = inner(v0, a )*dx

        if f is not None:
            L_1 += dt*inner(f, a)*dx

        #!SECTION MOMENTUM EQUATION

        # SECTION DIVERGENCE ZERO CONDITION
        # NOTE - see below directly in definition of the final variational form
        
        zero_h = Function(p0.function_space)
        L_2 = inner(zero_h, h) * dx

        #!SECTION DIVERGENCE ZERO CONDITION

        # SECTION DIRECTOR EQUATION
        a_33 = inner(d1, c)*dxL

        # Ericksen stress tensor
        a_31 = (-1)*dt*inner(dot(grad_d0 , v1), a_times_b_times_c(d_,d_, c))*dxL

        # Leslie stress tensor
        if args.beta != 0.0:
            a_31 += dt* args.beta *inner(dot(grad_skw(v1),d_), a_times_b_times_c(d_,d_,c))*dxL
        if args.lam != 0.0:
            a_31 -= dt* args.lam *inner(dot(grad_sym(v1),d_), a_times_b_times_c(d_,d_,c))*dxL

        if args.gamma != 0.0:
            a_34 = (-1)*dt* args.gamma *inner(q1, a_times_b_times_c(d_,d_,c))*dxL

        L_3 = inner(d0, c)*dxL

        #!SECTION DIRECTOR EQUATION

        # SECTION EQUATION FOR THE VARIATIONAL DERIVATIVE
        a_43 = q_elastic_energy(args, d1, d0, b, H = H)

        a_44 = (-1)* inner(q1, b)*dxL 

        zero_b = Function(q0.function_space)
        L_4 = inner(zero_b , b) * dx

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
        ]) 

        return a, L

def q_elastic_energy(args, d1, d0, b, H = None):
    eq = args.K1 * inner( grad(d1), grad(b))*dx
    if args.K2 != 0.0:
        eq += args.K2 * inner( div(d1), div(b))*dx
    if args.K3 != 0.0:
        eq += args.K3 * inner( curl(d1), curl(b))*dx
    if args.K4 != 0.0 or args.K5 != 0.0:
        raise ValueError("K4 != 0 or K5 != 0 is not supported in the linear scheme.")
        # eq += args.K4 * inner( d1, curl(d1)) * ( inner(d0, curl(b)) + inner(b, curl(d1)) )*dx
        #NOTE - Since the curl is present a simplification to 2D is not possible.
        # eq += args.K5 * inner( cross( d1, curl(d1)) , cross(d0, curl(b)) + cross(b, curl(d1)) )*dx
        #NOTE - The cross product could be replaced by an according tangential matrix. However, since the curl is present a simplification to 2D is not possible anyways.

    if H is not None and args.chi_vert != 0.0:
        eq -= args.chi_vert * inner( d1, H)*inner( b, H)*dx
    if H is not None and args.chi_perp != 0.0:
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
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)

    # Return the index sets representing the row and column spaces. 2 times Blocks spaces
    nested_IS = P.getNestISs()
    ksp.getPC().setFieldSplitIS(("u", nested_IS[0][0]), ("p", nested_IS[0][1]), ("d", nested_IS[0][2]), ("q", nested_IS[0][3]))

    # Set the preconditioners for each block. This is done via command line interface.
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

def compute_metrics(comm, args, v,p,d,d0,q, dxL, H= None, id =""):
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
    
    if H is not None:
        if args.chi_vert != 0.0:
            E_total -= args.chi_vert * E_H_vert 
        if args.chi_perp != 0.0:
            E_total -= args.chi_perp * E_H_perp 
    
    # DISSIPATION
    dt = args.dt
    dissipation_form = dt * (args.mu1 + (args.lam)**2 /args.gamma)*(inner(d0,dot(grad_sym(v),d0))*inner(d0,dot(grad_sym(v),d0))*dx )
    dissipation_form += dt * (args.mu5 + args.mu6 - (args.lam)**2 /args.gamma) * (inner( dot(grad_sym(v),d0), dot(grad_sym(v),d0))*dx)
    if args.sym_grad:
        dissipation_form += dt * args.mu4 *inner( grad_sym(v), grad_sym(v))*dx
    else:            
        dissipation_form += dt* args.mu4 *inner( grad(v), grad(v))*dx
    dissipation_form +=(-1)*dt* args.gamma *inner(q, a_times_b_times_c(d0,d0,q))*dxL

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
        "Eela1"+id : E_ela1,
        "Eela2"+id : E_ela2,
        "diss"+id  : dissipation,
        "orth"+id  : orthogonality,
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




