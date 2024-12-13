from time import process_time, ctime
from dolfinx.fem import Function, functionspace, dirichletbc, form, assemble_scalar, dirichletbc, locate_dofs_topological, locate_dofs_geometrical, ElementMetaData, Expression
from dolfinx.fem.petsc import LinearProblem, assemble_vector
from ufl import div, dx, grad, inner, VectorElement, FiniteElement, TensorElement, MixedElement, TrialFunctions, TrialFunction, TestFunction, split, Measure, lhs, rhs, FacetNormal, CellDiameter, jump, avg, ds, dS, dot, nabla_grad, cross, sqrt, sign, exp, as_ufl
from sim.models.el_forms import I_dd, grad_sym, grad_skw
from sim.common.common_fem_methods import *
from sim.common.error_computation import errornorm

# dolfinx v0.7

#SECTION - GENERAL METHOD
def linear_dg_method(experiment, args, projection: bool, postprocess=None, solver_metadata = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}):
    """
    Good options for the petsc metadata are:
        {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}  --> works mostly, very fast
        {"ksp_type": "gmres", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}    --> very stable
        {"ksp_type": "dgmres", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}   --> very stable
        {"ksp_type": "pgmres", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}   --> very stable
        {"ksp_type": "fgmres", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}   --> mostly stable 
        {"ksp_type": "fbcgs", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}    --> mostly stable
    """
   

    # SECTION PARAMETERS
    ## import and initalize parameters
    dim         = experiment.dim
    dt          = experiment.dt
    dh          = experiment.dh
    t           = experiment.t0
    t0          = experiment.t0
    T           = experiment.T
    const_A     = experiment.const_A
    v_el        = experiment.v_el
    mu_1        = experiment.mu_1
    mu_4        = experiment.mu_4
    mu_5        = experiment.mu_5
    mu_6        = experiment.mu_6
    lam         = experiment.lam    
    alpha       = args.alpha
    alphabc     = args.alpha
    #!SECTION PARAMETERS
    
    # SECTION MESH, MESHTAGS AND MEASURES
    domain      = experiment.mesh    
    meshtags    = experiment.meshtags
    dx          = Measure('dx', domain=domain) 
    ds          = Measure('ds', domain=domain, subdomain_data=meshtags) # since we allow for mix of boundary conditions                                         
    dS          = Measure('dS', domain=domain)
    n_F         = FacetNormal(domain)
    h_T         = CellDiameter(domain)
    #!SECTION


    submodel      = args.submod

    #add all local variables for transparency    
    postprocess.log("dict","static",{"model.vars":dict(locals())}, visible =False)

    initial_conditions          = experiment.initial_conditions
    boundary_conditions         = experiment.boundary_conditions

    computation_time  = 0
    last_time_measure = process_time()
    
    # SECTION - FUNCTION SPACES AND FUNCTIONS
    vP2         = VectorElement("Lagrange", domain.ufl_cell(), 2, dim = dim)    # for velocity field
    P1          = FiniteElement("Lagrange", domain.ufl_cell(), 1)               # for pressure
    vDG0        = VectorElement("DG", domain.ufl_cell(), 0, dim = dim)          # for director field and the discrete laplacian
    tDG0        = TensorElement("DG", domain.ufl_cell(), 0, shape = (dim,dim))  # for the discrete gradient
    me          = MixedElement([vP2, P1, vDG0, vDG0, tDG0])
    F           = functionspace(domain, me)
    
    v1,p1, d1, q1, psi1     = TrialFunctions(F)
    u0                      = Function(F)               # initial conditions
    v0,p0, d0, q0, psi0     = split(u0)
    u                       = Function(F)               # current iterate
    v,p, d, q, psi          = split(u)
    u_test                  = TestFunction(F)
    a,h, c, b, tau          = split(u_test)
    
    V, mapV = F.sub(0).collapse()
    Q, mapQ = F.sub(1).collapse()
    D, mapD = F.sub(2).collapse()
    Y, mapY = F.sub(3).collapse()  
    TF, mapY = F.sub(4).collapse()
    
    dbc = Function(D) # approximation of boundary condition
    d0_, d1_, d_diff = Function(D), Function(D), Function(D) # used later to compute the cell-wise orthogonality
        
    # output functions
    vDG1FS = functionspace(domain, VectorElement("DG", domain.ufl_cell(), 1, dim = dim))
    tDG1FS = functionspace(domain, TensorElement("DG", domain.ufl_cell(), 1, shape = (dim,dim)))
    v_out, p_out, d_out, q_out, psi_out = Function(V), Function(Q), Function(vDG1FS), Function(vDG1FS), Function(tDG1FS)

    #!SECTION FUNCTION SPACES AND FUNCTIONS
    
    # SECTION VARIATONAL FORMULATION
    postprocess.log("dict", "static",{"Status" : "Creating variational formulation"})

    

    # SECTION DEFINITION OF ENERGY
    Energy_kinetic      = 0.5*inner(v, v) *dx  
    Energy_elastic      = 0.5*inner(psi,psi)*dx
    Energy_J            = 0.5*(1/avg(h_T)) * inner( jump(d), jump(d) )*dS 
    Energy_bc           = 0.5*const_A*(1/h_T)* inner( d-dbc, d-dbc )*ds
    Energy_total        = Energy_kinetic + const_A*Energy_elastic +const_A*alpha*Energy_J + const_A*alpha*Energy_bc

    def compute_energies(appendix = ""):        
        E_ela   = assemble_scalar(form(Energy_elastic) )
        E_kin   = assemble_scalar(form(Energy_kinetic) )
        E_J     = assemble_scalar(form(Energy_J) )
        E_bc    = assemble_scalar(form(Energy_bc) )
        E_tot   = E_kin + const_A*E_ela +const_A*alpha*E_J + const_A*alpha*E_bc
        metrics = { "Energies (total)"+appendix: E_tot,
                    "Energies (elastic)"+appendix: E_ela,
                    "Energies (kinetic)"+appendix: E_kin,
                    "Energy int. jumps"+appendix: E_J, 
                    "Energy boundary cond."+appendix: E_bc
                    }
        return metrics
    #!SECTION

    # SECTION EQUATION SYSTEM DEFINITION

    def momentum_eq(v1,  p1, d0, q1, a):
        # discrete time derivative
        eq =  inner( v1 - v0 , a )*dx 

        # convection term
        eq += dt*inner(dot(v0, nabla_grad(v1)), a)*dx + dt*0.5*div(v0)*inner(v1, a)*dx 

        # pressure term and divergence zero condition
        eq += - dt*inner(p1,div(a))*dx + dt*div(v1)*h*dx 

        # diffusion term or dissipative terms
        if submodel == 2:
            eq += dt * mu_4*inner( grad_sym(v1), grad_sym(a))*dx
            eq += dt * v_el*(mu_1+lam**2)*inner(d0,dot(grad_sym(v1),d0))*inner(d0,dot(grad_sym(a),d0))*dx
            eq += dt * v_el* (mu_5+mu_6-lam**2)*inner( dot(grad_sym(v1),d0), dot(grad_sym(a),d0))*dx
        else:
            # standard diffusion case
            eq += dt* mu_4 *inner( grad(v1), grad(a))*dx

        # Ericksen stress tensor terms
        if dim ==3:
            eq += -dt*const_A*v_el*inner( cross(d0, dot(psi0 , a)),  cross(d0, q1))*dx
        else:
            eq += -dt*const_A*v_el*inner( dot( I_dd(d0, d0, dim) , dot(psi0 , a)),  q1)*dx

        # Leslie stress tensor terms
        if submodel == 2:
            if dim ==3:
                eq +=  -dt*lam*inner(cross( d0 , q1), cross(d0, dot(grad_sym(a), d0) ) )*dx
                eq += dt*inner(cross( d0 ,dot(grad_skw(a),d0)), cross(d0, q1))*dx
            else:
                eq +=  -dt*lam*inner(dot( I_dd(d0,d0, dim) , q1), dot(grad_sym(a), d0))*dx
                eq += dt*inner(dot( I_dd(d0,d0, dim) ,dot(grad_skw(a),q1)), d0)*dx

        return eq
    
    def discrete_laplacian_def(d_, psi_, q_, b_, dbc_):
        """
        This is the Equation defining the discrete Laplacian $q = \Delta_h d$ weakly for a given reconstructed gradient $\psi = \nabla_h d$.
        $$
            - (\Delta_h d ,b)_2 = (q,b)_2 = (R^0_h [d], R^0_h [b])
            = (\nabla_h d, \nabla_h [b])
        $$
        """
        # $q = - \Delta_h d$
        eq = -inner(q_,b_)*dx 

        # applying the Definition of the discrete lifting onto the discrete gradient of the test function b
        # NOTE - using the normal in direction '-' is consistent with the Definition of the discrete gradient and is necessary for the right results!
        eq += inner(dot(avg(psi_),n_F('-')),jump(b_))*dS      # on the interior

        # Penalization terms
        eq += (alpha/avg(h_T)) * inner( jump(d_), jump(b_) )*dS   # interior jump penalization 

        for bcwofs in boundary_conditions:
            if bcwofs.type == "Dirichlet" and bcwofs.quantity == "d":
                eq += -inner(dot(psi_, n_F), b_)*ds(bcwofs.meshtag)             # Definition of the discrete lifting on the boundary        
                eq += (alphabc/h_T)*inner(d_ - dbc_, b_)*ds(bcwofs.meshtag)     # bc penalization

        return eq

    def discrete_gradient_def(d_, psi_, tau_, dbc_):
        """
        This is the Equation defining the discrete lifting or reconstructed gradient for piecewise constant functions
            $$
            \psi = R^0 ([d]) = \nabla_h d
            $$
        """
        eq = -inner(psi_ , tau_)*dx
        # NOTE - using the normal in direction '-' is consistent with the Definition of the discrete gradient and is necessary for the right results!
        eq += inner(dot(avg(tau_),n_F('-')),jump(d_))*dS    # on the interior

        for bcwofs in boundary_conditions:
            if bcwofs.type == "Dirichlet" and bcwofs.quantity == "d":
                eq += inner(dot(tau_,n_F), dbc_ - d_ )*ds(bcwofs.meshtag)                # discrete gradient on the boundary with dirichlet bc

        return eq

    def director_eq( d1, d0, q1, v1, c):
        # discrete time derivative
        eq = inner(d1 - d0, c)*dx

        # dissipative terms
        if dim == 3: 
            eq += dt*const_A*inner( cross(d0, q1), cross(d0, c))*dx
        else: 
            eq += dt*const_A*inner(q1 , dot( I_dd(d0,d0, dim) , c) )*dx

        # Ericksen stress tensor terms
        if dim ==3:
            eq += dt*v_el*inner( cross(d0, dot(psi0 , v1)),  cross(d0, c))*dx
        else:
            eq += dt*v_el*inner( dot( I_dd(d0, d0, dim) , dot(psi0 , v1)),  c)*dx
   
        # Leslie stress tensor terms
        if submodel == 2:
            if dim ==3:
                eq +=  dt*lam*inner(cross( d0 , c), cross(d0, dot(grad_sym(v1), d0) ) )*dx
                eq += - dt*inner(cross( d0 ,dot(grad_skw(v1),d0)), cross(d0, c))*dx
            else:
                eq +=  dt*lam*inner(dot( I_dd(d0,d0, dim) , c), dot(grad_sym(v1), d0))*dx
                eq += - dt*inner(dot( I_dd(d0,d0, dim) ,dot(grad_skw(v1),c)), d0)*dx

        return eq
    
    #!SECTION
    
    Eq1 = momentum_eq(v1, p1, d0, q1, a)
    Eq2 = discrete_laplacian_def(d1, psi1, q1, b, dbc) + discrete_gradient_def(d1, psi1, tau, dbc)  
    Eq3 = director_eq( d1, d0, q1, v1, c)

    Form = Eq1 + Eq2 + Eq3
    A = lhs(Form)
    L = rhs(Form)

    #!SECTION VARIATONAL FORMULATION

    # SECTION INITIAL CONDITIONS 
    #TODO - Make this work if there is more than one value function passed.
    # retrieve boundary condition for placeholder function dbc
    for bcwofs in boundary_conditions:
            if bcwofs.type == "Dirichlet" and bcwofs.quantity == "d":
                dbc.interpolate(bcwofs.values)

    postprocess.log("dict","static",{"Status" : "Setting initial conditions"})
    u0.sub(0).interpolate(initial_conditions["v"]) 
    u0.sub(1).interpolate(initial_conditions["p"]) 
    u0.sub(2).interpolate(initial_conditions["d"]) 

    # COMPUTATION OF INITIAL LAPLACIAN AND DISCRETE GRADIENT    
    q01 = TrialFunction(Y)
    b01 = TestFunction(Y)
    psi01 = TrialFunction(TF)
    tau01 = TestFunction(TF)

    def discrete_grad_of(d_):
        psi_ic          = Function(TF)
        eqpsi0          = discrete_gradient_def(d_, psi01, tau01, dbc)  
        problem_psi0    = LinearProblem(lhs(eqpsi0), rhs(eqpsi0),  bcs=[], u=psi_ic, petsc_options=solver_metadata)
        problem_psi0.solve()
        return psi_ic
    u0.sub(4).interpolate(discrete_grad_of(d0))

    def discrete_laplacian_of(d_, psi_):
        q_ic            = Function(Y)
        eqq0            = discrete_laplacian_def(d_, psi_, q01, b01, dbc) 
        problem_q0      = LinearProblem(lhs(eqq0), rhs(eqq0),  bcs=[], u=q_ic, petsc_options=solver_metadata)
        problem_q0.solve()
        return q_ic
    u0.sub(3).interpolate(discrete_laplacian_of(d0, psi0))

    # OVERWRITING FUNCTION U
    u.x.array[:] = u0.x.array[:]
    #!SECTION

    #SECTION - BOUNDARY CONDITIONS
    postprocess.log("dict", "static",{"Status" : "Setting boundary conditions"})
    bcs = []
    
    for bcwofs in boundary_conditions:
        if bcwofs.type == "Dirichlet":
            # NOTE - Boundary conditions for d are enforced weakly
            if bcwofs.quantity == "v":
                bcwofs.set_fs((F.sub(0), V))
                bcs.append(bcwofs.bc)
        # elif bcwofs.type == "Neumann":            
        # elif bcwofs.type == "Robin":
        else: postprocess.log("dict", "static",{"Warning" : "Boundary conditions of type "+bcwofs.type+" are currently not implemented and will be ignored..."} )
    
    #!SECTION
    
    #SECTION - SETTING UP LINEAR PROBLEM
    postprocess.log("dict", "static",{"Status" : "Setting up linear problem"})
    problem_el = LinearProblem(A, L, bcs=bcs,  u=u, petsc_options= solver_metadata)   
    # problem_el.solver.setTolerances(rtol=1e-12, atol = 1e-15) #[rtol, atol, divtol, max_it]
    #!SECTION

    computation_time += process_time() - last_time_measure  

    #SECTION - POSTPROCESSING FOR t=0
    postprocess.log("dict", "static",{"Status" : "Postprocessing for t=0"})
    v_out.interpolate(u0.sub(0))
    p_out.interpolate(u0.sub(1))
    d_out.interpolate(u0.sub(2))
    q_out.interpolate(u0.sub(3))
    psi_out.interpolate(u0.sub(4))

    postprocess.log_functions(0.0, {"v": v_out, "p":p_out, "d":d_out, "q":q_out, "psi":psi_out}, mesh = domain, meshtags = meshtags)

    metrics = {"time":t}
    metrics.update(compute_energies())
    postprocess.log("dict", t, metrics)
    #!SECTION

    # SECTION TIME EVOLUTION
    postprocess.log("dict", "static",{"Status" : "Starting time evolution"})
    last_time_measure = process_time()

    while t < T:
        t += dt 
        
        postprocess.log("dict", t, {"Status" : "Solving linear problem", "time" : t, "progress" : (t-t0)/T })

        problem_el.solve()

        metrics = {"time":t}

        # SECTION - NODAL PROJECTION STEP      
        if projection:
            """
            The method Pi_h applies the nodal orthogonal projection:
            $ \mathcal{I}_h (\Tilde{d}^j_{o}/\abs{\Tilde{d}^j_{o}}) $
            with 
            $\Tilde{d}^j_{o} := (d^{j-1} + (I-d^{j-1}\otimes d^{j-1}) (\Tilde{d}^j-d^{j-1})) $
            """

            # SECTION METRICS BEFORE PROJECTION STEP
            metrics.update(compute_energies(appendix=" b4 projection"))
            d0_.interpolate(u0.sub(2))
            d1_.interpolate(u.sub(2))
            d_diff.x.array[:] = d1_.x.array[:]- d0_.x.array[:]
            tmp = {
                    "unit norm err b4 projection": test_unit_norm(u.sub(2).collapse()),                 
                    "nodal orthogonality (Linfty) b4 projection": test_ptw_orthogonality(d_diff, d0_),
                    }
            metrics.update(tmp)
            #!SECTION
            
            # ACTUAL PROJECTION STEP
            u.sub(2).interpolate(nodal_projection_unit(u.sub(2).collapse()))
            # update discrete gradient and laplacian
            u.sub(4).interpolate(discrete_grad_of(d))
            # u.sub(3).interpolate(discrete_laplacian_of(d, psi)) # not necessary, also not updated in convergence analysis
        #!SECTION 
        
        
        
        computation_time += process_time() - last_time_measure

        # SECTION - METRICS AFTER PROJECTION STEP        
        # Analytical errors
        errorL2, errorinf = "No exact solution available", "No exact solution available"
        if experiment.has_exact_solution:   
            errorL2 = experiment.compute_error(u.sub(2),t,norm = "L2", degree_raise = 3)   
            errorinf = experiment.compute_error(u.sub(2).collapse(),t,norm = "inf", degree_raise = 0)      
        
        # Collecting metrics
        metrics.update(compute_energies())
        d0_.interpolate(u0.sub(2))
        d1_.interpolate(u.sub(2))
        d_diff.x.array[:] = d1_.x.array[:]- d0_.x.array[:]
        tmp = {
                    "unit norm err": test_unit_norm(u.sub(2).collapse()),                 
                    "nodal orthogonality (Linfty)": test_ptw_orthogonality(d_diff, d0_),
                    "computation time": computation_time,
                    "errorL2": errorL2,
                    "errorinf": errorinf,
                    "datetime": str(ctime())}
        metrics.update(tmp)
        
        postprocess.log("dict", t, metrics)
        #!SECTION

       
        
        
        # SECTION - UPDATE AND POSTPROCESSING FUNCTIONS
        u0.x.array[:] = u.x.array[:]

        v_out.interpolate(u0.sub(0))
        p_out.interpolate(u0.sub(1))
        d_out.interpolate(u0.sub(2))
        q_out.interpolate(u0.sub(3))
        psi_out.interpolate(u0.sub(4))
        postprocess.log_functions(t, {"v": v_out, "p":p_out, "d":d_out, "q": q_out , "psi":psi_out}, mesh = domain, meshtags = meshtags)
        #!SECTION

    #!SECTION TIME EVOLUTION

    postprocess.close()

#!SECTION GENERAL METHOD


# SECTION NUMERICAL SCHEME BINDINGS
def lpdg(experiment, args, postprocess=None):
    """
    linear solver without projection and without mass-lumping
    """
    postprocess.log("dict", "static",{"model": "Linear DG projection scheme for the Ericksen-Leslie equations", "experiment":experiment.name})
    linear_dg_method(experiment, args, projection=True, postprocess = postprocess)

def ldg(experiment, args, postprocess=None):
    """
    linear solver without projection and without mass-lumping
    """
    postprocess.log("dict", "static",{"model": "Linear DG scheme for the Ericksen-Leslie equations", "experiment":experiment.name})
    linear_dg_method(experiment, args, projection=False, postprocess = postprocess)


#!SECTION BINDINGS


