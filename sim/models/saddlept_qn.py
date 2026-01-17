from ufl import grad, div, curl, split
from ufl import inner, dot, cross, derivative, dx, Measure, FacetNormal
from ufl import TrialFunction, TestFunction
from basix.ufl import element, mixed_element
import dolfinx.la as la
from dolfinx.fem import Function, functionspace, form, assemble_scalar
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_matrix, create_vector, set_bc, apply_lifting
from sim.common.operators import *
from sim.common.norms import L2_norm, H1_norm
from sim.common.common_fem_methods import *
from sim.common.meta_bcs import *


def saddlept_qn(comm, experiment, args, postprocess=None):
    # NOTE - not performing well currently..
    """
    Nodal implicit method (see Section 6) using a Quasi-Newton-scheme by Badia, S., Guillen-González, F., & Gutiérrez-Santacreu, J. (2011). Finite element approximation of nematic liquid crystal flows using a saddle-point structure. J. Comput. Phys., 230(4), 1686-1706.

    Important properties of the algorithm:
    - fulfills a discrete energy law
    - no auxiliary variable for the variational derivative of the energy needed
    - penalization variable for the unit-norm constraint

    Recommended settings:
    rtol        : 1e-6 (Choice of the paper)
    atol        : 1e-7
    ksp_type_u  : gmres
    pc_type_u   : ??
    ksp_type_d  : gmres
    pc_type_d   : ??
    """

    if postprocess and comm.rank == 0:
        postprocess.log("dict", "static", {"# MPI Ranks": comm.size})

    # SECTION PARAMETERS
    dim = args.dim
    dt  = args.dt
    t   = 0
    
    if postprocess and comm.rank == 0:
        postprocess.log("dict", "static",{"SID": args.sim_id, "MODEL":args.mod, "EPSILON/ALPHA": args.alpha, "MASS LUMPING" : args.mass_lumping, "EXPERIMENT":args.exp, "mesh res.": args.dh, "dt":args.dt, "dim":args.dim, "T": args.T, "max_it" : args.n_max_it, "atol" : args.atol, "rtol" : args.rtol, "ksp1" : args.ksp_type_u, "pc1": args.pc_type_u,"ksp2" : args.ksp_type_d, "pc2": args.pc_type_d})

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

    vP2         = element("Lagrange", domain.basix_cell(), 2, shape=(dim,))
    P1          = element("Lagrange", domain.basix_cell(), 1)
    vP1         = element("Lagrange", domain.basix_cell(), 1, shape=(dim,))  

    TH          = functionspace(domain, mixed_element([vP2, P1]))
    FS          = functionspace(domain, mixed_element([vP1, P1]))
    
    vp1, dxi1       = Function(TH), Function(FS)
    v1, p1          = split(vp1)
    d1, xi1         = split(dxi1)

    vp0, dxi0       = Function(TH), Function(FS)
    v0, p0          = split(vp1)
    d0, xi0         = split(dxi0)

    ah, cpsi        = TestFunction(TH), TestFunction(FS)
    a, h            = split(ah)
    c, psi          = split(cpsi)

    V, mapV = TH.sub(0).collapse()
    Q, mapQ = TH.sub(1).collapse()
    D, mapD = FS.sub(0).collapse()
    Y, mapY = FS.sub(1).collapse()     

    

    # COMPUTE AND SAVE DOFS PER RANK
   
    local_dofs = TH.dofmap.index_map.size_local + FS.dofmap.index_map.size_local  # Count DOFs on this rank    
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
    
    dxL = Measure("dx", domain = domain, metadata = {"quadrature_rule": "vertex", "quadrature_degree": 0})

    # SECTION INITIAL CONDITIONS
    vp0.sub(0).interpolate(initial_conditions["v"]) 
    vp0.sub(1).interpolate(initial_conditions["p"]) 
    dxi0.sub(0).interpolate(initial_conditions["d"]) 

    vp1.x.array[:] = vp0.x.array[:]
    dxi1.x.array[:] = dxi0.x.array[:]

    if "H" in initial_conditions.keys():
        H = Function(D)
        H.interpolate(initial_conditions["H"])
        # TODO - scatter forward
    else:
        H = None
    scatter_all([vp0, dxi0, vp1, dxi1])
    
    #!SECTION

    #SECTION - BOUNDARY CONDITIONS
    bcs_vp = []
    bcs_dxi = []
    
    p_dirichlet_bcs_exist = False # NOTE - this becomes relevant later for the assembly and solver setup to decide whether a nullspace needs to be prescribed

    for bcwofs in boundary_conditions:
        if bcwofs.type == "Dirichlet":
            if bcwofs.quantity == "v":
                bcwofs.set_fs((TH.sub(0), V))
                bcs_vp.append(bcwofs.bc)
            elif bcwofs.quantity == "p":
                bcwofs.set_fs((TH.sub(1), Q))
                bcs_vp.append(bcwofs.bc)
                p_dirichlet_bcs_exist = True
            elif bcwofs.quantity == "d":
                bcwofs.set_fs((FS.sub(0), D))
                bcs_dxi.append(bcwofs.bc)
                # NOTE - since we do not want stress on the boundary we prescribe homogeneous dirichlet bcs for the lagrange multiplier
                xi_zero = Function(Y)
                bcxi = meta_dirichletbc("xi", bcwofs.find_dofs, xi_zero, marker = bcwofs.marker , entity_dim = bcwofs.dim, entities = bcwofs.entities)
                bcxi.set_fs((FS.sub(1), Y))
                bcs_dxi.append(bcxi.bc)
        # elif bcwofs.type == "Neumann":            
        # elif bcwofs.type == "Robin":
        else: postprocess.log("dict", "static",{"Warning" : "Boundary conditions of type "+bcwofs.type+" are currently not implemented and will be ignored..."} )
    
    #!SECTION

    #SECTION - VARIATIONAL FORM
    a1, L1, a2, L2 = variational_form(dxL,  v1, p1, d1, xi1, v0, p0, d0, xi0, a, h, c, psi, dt, n_F, args, Q, Y, H=H, f= None, p_dirichlet_bcs_exist = p_dirichlet_bcs_exist)
    #!SECTION

    #SECTION - SETUP SOLVERS
    J1 = derivative(a1 - L1, vp1)
    J2 = derivative(a2 - L2, dxi1)
    
    jacobian1, jacobian2 = form(J1), form(J2)
    
    A1, A2 = create_matrix(jacobian1), create_matrix(jacobian2)
    b1, b2 = create_vector(form(a1 - L1)), create_vector(form(a2 - L2))

    solver1, solver2 = setup_solvers(comm, args, A1, A2)
    
    diff1, diff2 = Function(TH), Function(FS)
    
    #SECTION - POSTPROCESSING FOR t=0

    if postprocess:
        postprocess.log_functions(0.0, {"v": v0, "p":p0, "d":d0, "xi":xi0}, mesh = domain) #, meshtags = meshtags)

    metrics = compute_metrics(comm, args, vp0, dxi0, vp0, dxi0, dxL , H = H) # for initial condition
    if postprocess and comm.rank == 0:
        postprocess.log("dict", t, { "time" : t} |  metrics )

    #!SECTION

    # SECTION TIME EVOLUTION
    total_time = mpi_time(comm, start = total_time_start )

    while t < args.T:
        t += dt 

        # SECTION - ASSEMBLY AND SOLVING
        measure_start = mpi_time(comm)

        # SECTION - QUASI NEWTON METHOD
        i = 0
        converged = False
        v_divisor = vp0.x.petsc_vec.norm(0) # L2_norm(comm,v0)
        d_divisor = dxi0.x.petsc_vec.norm(0) # H1_norm(comm,d0)
        while (i < args.n_max_it) and not converged:
           
            # Assemble Jacobian and residual
            assemble_and_solve_system(solver2, b2, A2, jacobian2, form(a2-L2), dxi1, diff2, bcs=bcs_dxi)
            dxi1.x.array[:] += diff2.x.array
            dxi1.x.scatter_forward()
            
            # Assemble Jacobian and residual
            assemble_and_solve_system(solver1, b1, A1, jacobian1, form(a1-L1), vp1, diff1, bcs=bcs_vp)
            vp1.x.array[:] += diff1.x.array
            vp1.x.scatter_forward()

            i += 1

            # Compute norm of update
            v_norm = diff1.x.petsc_vec.norm(0) #L2_norm(comm,diff1.sub(0))
            d_norm = diff2.x.petsc_vec.norm(0) #H1_norm(comm,diff2.sub(0))

            if (v_norm <= args.atol or v_norm <= v_divisor * args.rtol) and (d_norm <= args.atol or d_norm <= d_divisor * args.rtol):
                converged = True
            if postprocess: 
                metrics =  compute_metrics(comm, args, vp1, dxi1, vp0, dxi0, dxL , H=H)
                postprocess.log("dict", t, { "time" : t , "n_iters" : i, "n_norm" : v_norm + d_norm, "converged": converged} | metrics )
                postprocess.log_message(f"v_norm: {v_divisor}")
                postprocess.log_message(f"d_norm: {d_divisor}")
                postprocess.log_message(f"v_diff_norm: {v_norm}")
                postprocess.log_message(f"d_diff_norm: {d_norm}")

        #!SECTION
        
        measure_time = mpi_time(comm, start= measure_start)

        #!SECTION
       

        
        #SECTION - METRICS AT END OF ITERATION 
        errorL2 = np.nan
        if experiment.has_exact_solution:   
            errorL2 = experiment.compute_error(comm, dxi1.sub(0).collapse(),t,norm = "L2", degree_raise = 3)   
                   

        
        total_time = mpi_time(comm, start = total_time_start )
        if postprocess and comm.rank == 0:
            # metrics =  compute_metrics(comm, args, vp1, dxi1, vp0, dxi0, dxL , H=H)
            postprocess.log("dict", t, { "time" : t, "errorL2" : errorL2 , "t.tot" : total_time}) # | metrics)
            postprocess.log("dict", t, {"time":t, "t.sol_ass": measure_time}, visible = True)  
        
        #!SECTION

        #SECTION - UPDATE
        update_and_scatter([vp0, dxi0], [vp1, dxi1])
        #!SECTION 

        #SECTION - SAVING
        if postprocess: 
            postprocess.log_functions(t, {"v": v0, "p":p0, "d":d0, "xi":xi0}, mesh = domain, meshtags = meshtags)

        #!SECTION

        assert converged

    #!SECTION TIME EVOLUTION

    if postprocess: 
        postprocess.close()



def assemble_and_solve_system(solver, b, A, jacobian, residual, uh, du, bcs=[]):
    """
    Based on:
    https://jsdokken.com/dolfinx-tutorial/chapter4/newton-solver.html
    """
    with b.localForm() as loc_b:
        loc_b.set(0)
    A.zeroEntries()
    assemble_matrix(A, jacobian, bcs = bcs)
    A.assemble()
    assemble_vector(b, residual)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    # Scale residual by -1
    b.scale(-1)

    # Apply Bcs
    apply_lifting(b, [jacobian], [bcs], x0=[uh.x.petsc_vec])
    set_bc(b, bcs, uh.x.petsc_vec, 1.0)
    b.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    # Solve linear problem
    solver.solve(b, du.x.petsc_vec)
    du.x.scatter_forward()

    return b

def variational_form(dxL,  v1, p1, d1, xi1, v0, p0, d0, xi0, a, h, c, psi, dt, normal_F, args, Q, Y,  H=None, f= None, p_dirichlet_bcs_exist = False):
        """
        The variational form is structured as followed:
        a(.,.) : bilinear form describing the lhs of the system
        L(.) : linear form describing the rhs of the system
        a_ij(.,.) : bilinear form describing the lhs of the system depending on the i-th test function and the j-th trial function
        e.g.
        a_11 --> depends on v1 and a
        a_14 --> depends on q1 and a
        """
        v_ = 0.5*v1 + 0.5*v0
        d_ = 0.5*d1 + 0.5*d0

        # SECTION MOMENTUM EQUATION
        a_11 = inner( v1 , a )*dx # discrete time derivative

        a_11 += dt* ( inner(dot(v_, nabla_grad(v_)), a) + 0.5*div(v_)*inner(v_, a)  )*dx # temam's convection term

        if args.mu4 != 0.0:
            if args.sym_grad:
                a_11 += dt * args.mu4 *inner( grad_sym(v_), grad_sym(a))*dx
            else:            
                a_11 += dt* args.mu4 *inner( grad(v_), grad(a))*dx

        coeff1 = args.mu1 + (args.lam)**2 /args.gamma
        if coeff1 != 0.0:
            a_11 += dt * coeff1*inner(d0,dot(grad_sym(v_),d0))*inner(d0,dot(grad_sym(a),d0))*dx

        coeff2 = args.mu5 + args.mu6 - (args.lam)**2 /args.gamma
        if coeff2 != 0.0:
            a_11 += dt * coeff2 *inner( dot(grad_sym(v_),d0), dot(grad_sym(a),d0))*dx

        # Reformulated pressure term
        a_12 = dt*(-1)*inner(p1, div(a)) * dx 
        if p_dirichlet_bcs_exist:
            a_12 += dt*inner(p1, dot(a, normal_F)) * ds
            # NOTE - this is not the most elegant implementation
            # On the parts of the boundary where we strongly enforce the dirichlet bcs for the pressure, this will simply turn into a prescribed forcing term during assembly.
            # On the parts where this is not the case and further no-slip bcs for the velocity are employed, this term will simply vanish during assembly. 
            # This could be made more efficient by case distinction etc
        
        L_1 = inner(v0, a )*dx
        # Ericksen stress tensor
        # Crank-Nicholson discretization
        L_1 += 0.5*dt*q_elastic_energy(args, d1, d0, dot(grad(d0) , a), H = H)
        L_1 += 0.5*dt*q_elastic_energy(args, d0, d0, dot(grad(d0) , a), H = H)

        # Leslie stress tensor
        if args.beta != 0.0:            
            # Crank-Nicholson discretization
            L_1 -= 0.5*dt* args.beta *q_elastic_energy(args, d1, d0, dot(grad_skw(a),d_), H = H)
            L_1 -= 0.5*dt* args.beta *q_elastic_energy(args, d0, d0, dot(grad_skw(a),d_), H = H)
        if args.lam != 0.0:            
            # Crank-Nicholson discretization
            L_1 += 0.5*dt* args.lam *q_elastic_energy(args, d1, d0, dot(grad_sym(a),d_), H = H)
            L_1 += 0.5*dt* args.lam *q_elastic_energy(args, d0, d0, dot(grad_sym(a),d_), H = H)

        if f is not None:
            L_1 += dt*inner(f, a)*dx

        #!SECTION MOMENTUM EQUATION

        # SECTION DIVERGENCE ZERO CONDITION
        # NOTE - see below directly in definition of the final variational form
        
        # zero_h = Function(Q)
        # L_2 = inner(zero_h, h) * dx

        #!SECTION DIVERGENCE ZERO CONDITION

        # SECTION DIRECTOR EQUATION
        a_33 = inner(d1, c)*dx

        # Ericksen stress tensor
        a_33 += dt*inner(dot(grad(d0) , v_), c)*dx

        # Leslie stress tensor
        if args.beta != 0.0:
            a_33 -= dt* args.beta *inner(dot(grad_skw(v_),d_), c)*dx
        if args.lam != 0.0:
            a_33 += dt* args.lam *inner(dot(grad_sym(v_),d_), c)*dx

        if args.gamma != 0.0:
            # Crank-Nicholson discretization
            a_33 += 0.5*dt* args.gamma * q_elastic_energy(args, d1, d0, c, H = H)
            a_33 += 0.5*dt* args.gamma * q_elastic_energy(args, d0, d0, c, H = H)

        a_34 = dt*args.gamma*0.5*(xi1+xi0)*inner(d_,c)*dxL

        L_3 = inner(d0, c)*dxL

        #!SECTION DIRECTOR EQUATION

        # SECTION EQUATION FOR THE VARIATIONAL DERIVATIVE
        a_43 = psi*inner(d1,d1)*dxL

        if args.alpha > 0.0:
            a_44 = (-1)*(args.alpha**2)* xi1 * psi *dxL
        else:
            xi_zero = Function(Y)
            a_44 = (-1)*xi_zero* xi1 * psi *dxL


        L_4 = 1.0 * psi *dxL

        #!SECTION EQUATION FOR THE VARIATIONAL DERIVATIVE
        
        """
        Full bilinear form (lhs) and linear form (rhs)
        """
        # zero_block_p =  zero_h * p1 * h * dx if p_dirichlet_bcs_exist else None # NOTE - the zero block is assembled in the case of Dirichlet boundary conditions for the pressure
        a1 = a_11 + a_12
        a1 -= inner(div(v1), h) * dx

        # a1 = form([
        #     [a_11, a_12], 
        #     [(-1)*inner(div(v1), h) * dx, zero_block_p ],
        # ])

        L1 = L_1 # L_2 is now a None object
        # L1 = form([
        #     L_1 ,
        #     L_2 ,
        # ]) 

        a2  = a_33 + a_34
        a2 += a_43 + a_44
        # a2 = form([
        #     [a_33, a_34], 
        #     [a_43, a_44]
        # ])

        L2 = L_3 + L_4
        # L2 = form([
        #     L_3 ,
        #     L_4 ,
        # ]) 

        return a1, L1, a2, L2

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

    if not (H is None) and args.chi_vert != 0.0:
        eq -= args.chi_vert * inner( d1, H)*inner( b, H)*dx
    if not (H is None) and args.chi_perp != 0.0:
        eq += args.chi_perp * inner(  H, a_times_b_times_c(d1, b, H) )*dx

    return eq

def setup_solvers(comm, args, A1, A2):
    ksp1, ksp2 = PETSc.KSP().create(comm), PETSc.KSP().create(comm)
    
    ksp1.setOperators(A1)
    ksp2.setOperators(A2)

    opts = PETSc.Options()
    option_prefix = ksp1.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = args.ksp_type_u
    opts[f"{option_prefix}pc_type"] = args.pc_type_u
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = args.pc_factor_mat_solver_type
    opts[f"{option_prefix}ksp_atol"] = args.ksp_atol 
    opts[f"{option_prefix}ksp_rtol"] = args.ksp_rtol 
    ksp1.setFromOptions()

    opts = PETSc.Options()
    option_prefix = ksp2.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = args.ksp_type_d
    opts[f"{option_prefix}pc_type"] = args.pc_type_d
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = args.pc_factor_mat_solver_type
    opts[f"{option_prefix}ksp_atol"] = args.ksp_atol 
    opts[f"{option_prefix}ksp_rtol"] = args.ksp_rtol 
    ksp2.setFromOptions()

    return ksp1, ksp2

def compute_metrics(comm, args, vp1, dxi1, vp0, dxi0, dxL, H= None, id =""):
    v,p,d,q = vp1.sub(0), vp1.sub(1), dxi1.sub(0), dxi1.sub(1)
    d0 = dxi0.sub(0)
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
