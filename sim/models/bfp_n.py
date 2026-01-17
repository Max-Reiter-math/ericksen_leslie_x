from ufl import grad, div, curl, inner, dot, cross, dx, FacetNormal, TestFunction, split
from basix.ufl import element, mixed_element
from dolfinx import log
from dolfinx.fem import Function, functionspace, form, assemble_scalar, ElementMetaData
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from sim.common.operators import *
from sim.common.common_fem_methods import *
from sim.common.meta_bcs import *


def bfp_n(comm, experiment, args, postprocess=None):
    """
    Algorithm 4.1 by Becker, R., Feng, X., & Prohl, A. (2008). Finite element approximations of the Ericksen-Leslie model for nematic liquid crystal flow. SIAM Journal on Numerical Analysis, 46(4), 1704-1731. doi:10.1137/07068254X

    Implemented using a monolithic Newton solver.

    Important properties of the algorithm:
    - fulfills discrete energy law
    - existence of solutions has been proven under a mesh constraint
    - Newton's method for the solution of the nonlinear system at each time step is used. The stopping criterion used is to require the L2-norm of the residual to be less than 1e-10
    - asymptotic fulfillment of the unit-norm constraint under time-step restriction, see Thm. 4.4

    Recommended settings for the Krylov solver:
    ksp_type_n : preonly
    pc_type_n : lu
    These settings lead to the three iterations per time-step as described in the paper for the experiment _smooth_.
    """

    if postprocess and comm.rank == 0:
        postprocess.log("dict", "static", {"# MPI Ranks": comm.size})

    # SECTION PARAMETERS
    dim = args.dim
    dt  = args.dt
    t   = 0
    
    if postprocess and comm.rank == 0:
        postprocess.log("dict", "static",{"SID": args.sim_id, "MODEL":args.mod,  "EXPERIMENT":args.exp, "mesh res.": args.dh, "dt":args.dt, "dim":args.dim, "T": args.T, "ksp": args.ksp_type_n, "pc" : args.pc_type_n})

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

    # SECTION - FUNCTION SPACES AND FUNCTIONS
    vP2         = element("Lagrange", domain.basix_cell(), 2, shape=(dim,))
    P1          = element("Lagrange", domain.basix_cell(), 1)
    vP1         = element("Lagrange", domain.basix_cell(), 1, shape=(dim,))
    me          = mixed_element([vP2, P1, vP1, vP1])
    FS          = functionspace(domain, me)
    TensorF     = functionspace(domain, ElementMetaData("Lagrange", 1 , shape=(dim, dim)))
    
    u1              = Function(FS)
    v1,p1, d1, q1   = split(u1)
    u0              = Function(FS)               # initial conditions
    v0,p0, d0, q0   = split(u0)
    u_test          = TestFunction(FS)
    a,h, c, b       = split(u_test)
    
    V, mapV = FS.sub(0).collapse()
    Q, mapQ = FS.sub(1).collapse()
    D, mapD = FS.sub(2).collapse()
    Y, mapY = FS.sub(3).collapse()       
    
    # FOR COMPUTATION OF INITIAL q
    qc1 = TrialFunction(Y)
    q_ic = Function(Y)
    bc_test = TestFunction(Y)  

    # COMPUTE AND SAVE DOFS PER RANK
   
    local_dofs = FS.dofmap.index_map.size_local    # Count DOFs on this rank    
    local_info = {f"rank {comm.rank} dofs": local_dofs}                                                                                           # Create local dictionary   
    all_info = comm.gather(local_info, root=0)                                                                                                      # Gather all dictionaries at root

    if comm.rank == 0:
        # Merge all into a single dict
        combined = {}
        for partial_dict in all_info:
            combined.update(partial_dict)
        if not (postprocess is None):
            postprocess.log("dict", "static", combined)

    #!SECTION FUNCTION SPACES AND FUNCTIONS
    
    # SECTION VARIATONAL FORMULATION

    # SECTION INITIAL CONDITIONS
    u0.sub(0).interpolate(initial_conditions["v"]) 
    u0.sub(1).interpolate(initial_conditions["p"]) 
    u0.sub(2).interpolate(initial_conditions["d"]) 

    if "H" in initial_conditions.keys():
        H = Function(D)
        H.interpolate(initial_conditions["H"])
        # TODO - scatter forward
    else:
        H = None
       
    # COMPUTATION OF INITIAL DIVERGENCE (BECAUSE WE USE IT TO SET THE BOUNDARY CONDITIONS FOR q)
    problem0 = LinearProblem(form(inner(qc1, bc_test)*dx), form(q_elastic_energy(args, d0, d0, bc_test, H = None)),  bcs=[], u=q_ic, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
    problem0.solve()
    u0.sub(3).interpolate(q_ic)
    
    # INITIAL GUESS FOR NONLINEAR SOLVER
    u0.x.scatter_forward()
    u1.x.array[:] = u0.x.array[:]
    u1.x.scatter_forward()
    #!SECTION

    #SECTION - BOUNDARY CONDITIONS
    bcs = []
    
    p_dirichlet_bcs_exist = False # NOTE - this becomes relevant later for the assembly and solver setup to decide whether a nullspace needs to be prescribed

    for bcwofs in boundary_conditions:
        if bcwofs.type == "Dirichlet":
            if bcwofs.quantity == "v":
                bcwofs.set_fs((FS.sub(0), V))
                bcs.append(bcwofs.bc)
            elif bcwofs.quantity == "p":
                bcwofs.set_fs((FS.sub(1), Q))
                bcs.append(bcwofs.bc)
                p_dirichlet_bcs_exist = True
            elif bcwofs.quantity == "d":
                bcwofs.set_fs((FS.sub(2), D))
                bcs.append(bcwofs.bc)
                # BOUNDARY CONDITIONS FOR AUXILIARY VARIABLE
                #NOTE - We assume that the initial condition for the director field fulfills the boundary conditions imposed
                bcq = meta_dirichletbc("q", bcwofs.find_dofs, q_ic, marker = bcwofs.marker , entity_dim = bcwofs.dim, entities = bcwofs.entities)
                bcq.set_fs((FS.sub(3), Y))
                bcs.append(bcq.bc)
        # elif bcwofs.type == "Neumann":            
        # elif bcwofs.type == "Robin":
        else: postprocess.log("dict", "static",{"Warning" : "Boundary conditions of type "+bcwofs.type+" are currently not implemented and will be ignored..."} )
    
    #!SECTION

    #SECTION - VARIATIONAL FORM AND NONLINEAR PROBLEM
    a, L            = variational_form( v1, p1, d1, q1, v0, p0, d0, q0, a, h, b, c, dt, n_F, args, H=H, f= None, p_dirichlet_bcs_exist = p_dirichlet_bcs_exist)
    problem_nonlin  = NonlinearProblem(a-L, u1, bcs=bcs)
    #!SECTION

    #SECTION - SOLVER SETUP
    solver = NewtonSolver(MPI.COMM_WORLD, problem_nonlin)
    solver.report = False #True
    # log.set_log_level(log.LogLevel.INFO)
    #SECTION - EVTL. FINETUNING OF THE NONLINEAR SOLVER
    solver.convergence_criterion = "residual"   # NOTE - this choice is based on the paper
    solver.atol = 1e-10                         # NOTE - this choice is based on the paper
    solver.rtol = 1e-10                         # NOTE - this choice is based on the paper
    solver.max_it = args.n_max_it
    solver.error_on_nonconvergence = False
    # solver.report = True
    # log.set_log_level(log.LogLevel.INFO)
    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = args.ksp_type_n
    opts[f"{option_prefix}pc_type"] = args.pc_type_n
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = args.pc_factor_mat_solver_type
    opts[f"{option_prefix}ksp_atol"] = args.ksp_atol 
    opts[f"{option_prefix}ksp_rtol"] = args.ksp_rtol 
    ksp.setFromOptions()
    #!SECTION

    #SECTION - POSTPROCESSING FOR t=0

    if postprocess:
        postprocess.log_functions(0.0, {"v": v0, "p":p0, "d":d0, "q":q0}, mesh = domain) #, meshtags = meshtags)

    metrics = compute_metrics(comm, args, u0, u0, H = H) # for initial condition
    if postprocess and comm.rank == 0:
        postprocess.log("dict", t, { "time" : t} |  metrics )

    #!SECTION

    # SECTION TIME EVOLUTION
    total_time = mpi_time(comm, start = total_time_start )

    while t < args.T:
        t += dt 

        # SECTION - ASSEMBLY AND SOLVING
        measure_start = mpi_time(comm)

        newton_iterations, converged = solver.solve(u1)
        
        measure_time = mpi_time(comm, start= measure_start)

        #!SECTION

        #SECTION - UPDATE
        u1.x.scatter_forward()
        u0.x.array[:] = u1.x.array[:]
        u0.x.scatter_forward()
        #!SECTION 

        
        
        #SECTION - METRICS AT END OF ITERATION 
        errorL2 = np.nan
        if experiment.has_exact_solution:   
            errorL2 = experiment.compute_error(comm, u1.sub(2).collapse(),t,norm = "L2", degree_raise = 3)   

        metrics =  compute_metrics(comm, args, u1, u0, H=H)
        total_time = mpi_time(comm, start = total_time_start )
        if postprocess and comm.rank == 0:
            postprocess.log("dict", t, { "time" : t, "n_iters" : newton_iterations, "converged" :converged , "errorL2" : errorL2 , "t.tot" : total_time, "t.ass_sol" : measure_time} | metrics)
        
        #!SECTION

        #SECTION - SAVING
        if postprocess: 
            postprocess.log_functions(t, {"v": v0, "p":p0, "d":d0, "q":q0}, mesh = domain, meshtags = meshtags)

        #!SECTION

        assert (converged)

    #!SECTION TIME EVOLUTION

    if postprocess: 
        postprocess.close()

#!SECTION GENERAL METHOD



def variational_form( v1, p1, d1, q1, v0, p0, d0, q0, a, h, b, c, dt, normal_F, args, H=None, f= None, p_dirichlet_bcs_exist = False):
        """
        The variational form is structured as followed:
        a(.,.) : bilinear form describing the lhs of the system
        L(.) : linear form describing the rhs of the system
        a_ij(.,.) : bilinear form describing the lhs of the system depending on the i-th test function and the j-th trial function
        e.g.
        a_11 --> depends on v1 and a
        a_14 --> depends on q1 and a
        """

        d_ = 0.5*d1 + 0.5*d0

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
            a_11 += dt * coeff1*inner(d0,dot(grad_sym(v1),d0))*inner(d_,dot(grad_sym(a),d0))*dx

        coeff2 = args.mu5 + args.mu6 - (args.lam)**2 /args.gamma
        if coeff2 != 0.0:
            a_11 += dt * coeff2 *inner( dot(grad_sym(v1),d0), dot(grad_sym(a),d0))*dx

        # Reformulated pressure term
        a_12 = dt*(-1)*inner(p1, div(a)) * dx 
        if p_dirichlet_bcs_exist:
            a_12 += dt*inner(p1, dot(a, normal_F)) * ds
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

        if not (f is None):
            L_1 += dt*inner(f, a)*dx

        #!SECTION MOMENTUM EQUATION

        # SECTION DIVERGENCE ZERO CONDITION
        # NOTE - see below directly in definition of the final variational form    
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
        a_43 = 0.5*q_elastic_energy(args, d1, d0, b, H = H) + 0.5*q_elastic_energy(args, d0, d0, b, H = H)
        # NOTE - this is not exact in this framework since the known part of d_ should go to the rhs

        a_44 = (-1)* inner(q1, b)*dx 

        #!SECTION EQUATION FOR THE VARIATIONAL DERIVATIVE
        
        """
        Full bilinear form (lhs) and linear form (rhs)
        
        a = form([
            [a_11, a_12, None, a_14], 
            [(-1)*inner(div(v1), h) * dx, None, None, None],
            [a_31, None, a_33, a_34], 
            [None, None, a_43, a_44]
        ])

        L = form([
            L_1 ,
            None ,
            L_3 ,
            None ,
        ]) 
        """
        a  = a_11 + a_12 + a_14 
        a -= inner(div(v1), h) * dx
        a += a_31 + a_33 + a_34
        a += a_43 + a_44

        L = L_1 + L_3

        return a, L

def q_elastic_energy(args, d1, d0, b, H = None):
    eq = args.K1 * inner( grad(d1), grad(b))*dx
    if args.K2 != 0.0:
        eq += args.K2 * inner( div(d1), div(b))*dx
    if args.K3 != 0.0:
        eq += args.K3 * inner( curl(d1), curl(b))*dx
    if args.K4 != 0.0 or args.K5 != 0.0:
        eq += args.K4 * inner( d1, curl(d1)) * ( inner(d0, curl(b)) + inner(b, curl(d1)) )*dx
        #NOTE - Since the curl is present a simplification to 2D is not possible.
        eq += args.K5 * inner( cross( d1, curl(d1)) , cross(d0, curl(b)) + cross(b, curl(d1)) )*dx
        #NOTE - The cross product could be replaced by an according tangential matrix. However, since the curl is present a simplification to 2D is not possible anyways.

    if H is not None and args.chi_vert != 0.0:
        eq -= args.chi_vert * inner( d1, H)*inner( b, H)*dx
    if H is not None and args.chi_perp != 0.0:
        eq += args.chi_perp * inner(  H, a_times_b_times_c(d1, b, H) )*dx

    return eq

def compute_metrics(comm, args, u1, u0, H= None, id =""):
    v,p,d,q = u1.sub(0), u1.sub(1), u1.sub(2), u1.sub(3)
    d0 = u0.sub(2)
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
    d_mid = 0.5*d0 + 0.5*d
    dissipation_form +=(-1)*dt* args.gamma *inner(q, a_times_b_times_c(d_mid,d_mid,q))*dx

    dissipation = assemble_scalar(form( dissipation_form ))
    dissipation = comm.allreduce(dissipation, op=MPI.SUM)

    # NODAL UNIT-NORM AND ORTHOGONALITY
    d_ = d.collapse()
    d0_ = d0.collapse()
    orthogonality = np.max(np.abs( np.sum( (np.reshape( d_.x.array[:] , (-1, args.dim)) - np.reshape( d0_.x.array[:] , (-1, args.dim))) * np.reshape( d0_.x.array[:] , (-1, args.dim)) , axis=1 ) ))
    orthogonality = comm.allreduce(orthogonality, op=MPI.MAX)
    unit1 = np.max(np.linalg.norm(np.reshape( d_.x.array[:] , (-1, args.dim)), axis=1))    
    unit2 = np.min(np.linalg.norm(np.reshape( d_.x.array[:] , (-1, args.dim)), axis=1))
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




