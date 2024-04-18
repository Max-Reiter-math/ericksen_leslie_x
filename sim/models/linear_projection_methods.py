from time import process_time, ctime
from dolfinx.fem import Function, functionspace, dirichletbc, form, assemble_scalar, dirichletbc, locate_dofs_topological, locate_dofs_geometrical, ElementMetaData, Expression
from dolfinx.fem.petsc import LinearProblem, assemble_vector
from ufl import div, dx, grad, inner, VectorElement, FiniteElement, MixedElement, TrialFunctions, TrialFunction, TestFunction, split, Measure, lhs, rhs, FacetNormal, sqrt, dP, sign, exp, as_ufl
from sim.models.el_forms import *
from sim.common.common_fem_methods import *
from sim.common.error_computation import errornorm

# dolfinx v0.7

#SECTION - GENERAL METHOD
def linear_method(experiment, args, projection: bool, use_mass_lumping: bool, postprocess=None, solver_metadata =  {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}):
   

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
    #!SECTION PARAMETERS

    domain        = experiment.mesh    
    meshtags      = experiment.meshtags
    submodel      = args.submod

    #add all local variables for transparency    
    postprocess.log("dict","static",{"model.vars":dict(locals())}, visible =False)

    initial_conditions          = experiment.initial_conditions
    boundary_conditions         = experiment.boundary_conditions

    computation_time  = 0
    last_time_measure = process_time()
    
    # SECTION - FUNCTION SPACES AND FUNCTIONS
    vP2         = VectorElement("Lagrange", domain.ufl_cell(), 2, dim = dim)
    P1          = FiniteElement("Lagrange", domain.ufl_cell(), 1)
    vP1         = VectorElement("Lagrange", domain.ufl_cell(), 1, dim = dim)
    me          = MixedElement([vP2, P1, vP1, vP1])
    F           = functionspace(domain, me)
    TensorF     = functionspace(domain, ElementMetaData("Lagrange", 1 , shape=(dim, dim)))
    
    v1,p1, d1, q1   = TrialFunctions(F)
    u0              = Function(F)               # initial conditions
    v0,p0, d0, q0   = split(u0)
    u               = Function(F)               # current iterate
    v,p, d, q       = split(u)
    u_test          = TestFunction(F)
    a,h, c, b       = split(u_test)
    grad_d0_project = Function(TensorF)         # gradient of d0 projected onto P1
    
    V, mapV = F.sub(0).collapse()
    Q, mapQ = F.sub(1).collapse()
    D, mapD = F.sub(2).collapse()
    Y, mapY = F.sub(3).collapse()  
    
    # output functions
    v_out, p_out, d_out, q_out = Function(V), Function(Q), Function(D), Function(Y)

    qc1 = TrialFunction(Y)
    q_ic = Function(Y)
    bc_test = TestFunction(Y)  
    #!SECTION FUNCTION SPACES AND FUNCTIONS
    
    # SECTION VARIATONAL FORMULATION
    postprocess.log("dict", "static",{"Status" : "Creating variational formulation"})
    # Defining mass lumping
    dml = Measure("dx", domain = domain, metadata = {"quadrature_rule": "vertex", "quadrature_degree": 0}) # needed for error computatoin
    if use_mass_lumping:
        dxL = dml
    else:
        dxL = dx

    # Define Energies
    Energy_kinetic = 0.5*inner(v, v) *dx  
    Energy_elastic = 0.5*const_A*inner( grad(d ), grad(d ))*dx 
    Energy_total = Energy_kinetic + Energy_elastic
    
    # Momentum equation
    def momentum_eq(v1,  p1, d0, q1, a):
        eq =  inner( (v1-v0) , a )*dx 
        eq += dt*Convection_Velocity_Temam( v1, v0, a) *dx 
        eq += - dt*inner(p1,div(a))*dx + div(v1)*h*dx 
        if use_mass_lumping:
            eq += - dt*v_el* T_E(d0, d0, grad_d0_project, q1, a, dim, submodel = submodel)*dxL
        else:
            eq += - dt*v_el* T_E(d0, d0, grad(d0), q1, a, dim, submodel = submodel)*dx
        eq += dt*T_D(mu_1, mu_4, mu_5, mu_6, lam, v_el, d0,  v1, a, dim, submodel = submodel)*dx
        TL = T_L( lam, d0, d0, d0, q1, a, dim, submodel = submodel)
        if TL != None: 
            eq += dt*TL*dxL  
        return eq
    
    Eq1 = momentum_eq(v1, p1, d0, q1, a)

    # equation for the variational derivative, here discrete divergence (q = \Delta_h d)
    def discr_energy_eq(d, q, b):
        return const_A*inner( grad(d), grad(b))*dx - inner(q,b)*dxL 
        
    Eq2 = discr_energy_eq(d1, q1, b)  

    # director equation
    def director_eq( d1, d0, q1, v1, c):
        eq = inner(d1 - d0, c)*dxL
        if use_mass_lumping:
            eq += dt*v_el*T_E(d0, d0, grad_d0_project, c, v1, dim, submodel = submodel)*dxL
        else:
            eq += dt*v_el*T_E(d0, d0, grad(d0), c, v1, dim, submodel = submodel)*dxL
        eq += dt*D_D(d0, d0,q1, c, dim, submodel = submodel)*dxL 
        TL = T_L( lam, d0, d0, d0, c, v1, dim, submodel = submodel)
        if TL != None: eq += - dt*TL*dxL    
        return eq
    
    Eq3 = director_eq( d1, d0, q1, v1, c)

    Form = Eq1 + Eq2 + Eq3 
    A = lhs(Form)
    L = rhs(Form)

    #!SECTION VARIATONAL FORMULATION

    # SECTION INITIAL CONDITIONS
    postprocess.log("dict","static",{"Status" : "Setting initial conditions"})
    u0.sub(0).interpolate(initial_conditions["v"]) 
    u0.sub(1).interpolate(initial_conditions["p"]) 
    u0.sub(2).interpolate(initial_conditions["d"]) 
    if use_mass_lumping: 
        grad_d0_project.interpolate(project_lumped(grad(d0),TensorF))
    
    # COMPUTATION OF INITIAL DIVERGENCE
    eq0 = discr_energy_eq(d0, qc1, bc_test)  
    problem0 = LinearProblem(lhs(eq0), rhs(eq0),  bcs=[], u=q_ic, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
    problem0.solve()
    u0.sub(3).interpolate(q_ic)
    # OVERWRITING FUNCTION U
    u.x.array[:] = u0.x.array[:]
    #!SECTION

    #SECTION - BOUNDARY CONDITIONS
    postprocess.log("dict", "static",{"Status" : "Setting boundary conditions"})
    bcs = []
    
    for bcwofs in boundary_conditions:
        if bcwofs.type == "Dirichlet":
            if bcwofs.quantity == "v":
                bcwofs.set_fs((F.sub(0), V))
                bcs.append(bcwofs.bc)
            elif bcwofs.quantity == "d":
                bcwofs.set_fs((F.sub(2), D))
                bcs.append(bcwofs.bc)
                # BOUNDARY CONDITIONS FOR AUXILIARY VARIABLE
                #NOTE - We assume that the initial condition for the director field fulfills the boundary conditions imposed
                from sim.experiments.bcs_wo_fs import dirichlet_bc_wo_fs
                bcq = dirichlet_bc_wo_fs("q", bcwofs.find_dofs, q_ic, marker = bcwofs.marker , entity_dim = bcwofs.dim, entities = bcwofs.entities)
                # bcq = deepcopy(bcwofs)
                # bcq.values = q_ic
                bcq.set_fs((F.sub(2), D))
                bcs.append(bcwofs.bc)
                # bcs += [dirichletbc(q_ic, locate_dofs_geometrical((F.sub(3),Y), experiment.boundary))]
        # elif bcwofs.type == "Neumann":            
        # elif bcwofs.type == "Robin":
        else: postprocess.log("dict", "static",{"Warning" : "Boundary conditions of type "+bcwofs.type+" are currently not implemented and will be ignored..."} )
    
    #!SECTION
    
    #SECTION - SETTING UP LINEAR PROBLEM
    problem_el = LinearProblem(A, L, bcs=bcs,  u=u, petsc_options= solver_metadata)   
    # problem_el.solver.setTolerances(rtol=1e-12, atol = 1e-15) #[rtol, atol, divtol, max_it]
    #!SECTION

    computation_time += process_time() - last_time_measure  

    
    

    #SECTION - POSTPROCESSING FOR t=0
    v_out.interpolate(u0.sub(0))
    p_out.interpolate(u0.sub(1))
    d_out.interpolate(u0.sub(2))
    q_out.interpolate(u0.sub(3))
    postprocess.log_functions(0.0, {"v": v_out, "p":p_out, "d":d_out, "q":q_out}, mesh = domain) #, meshtags = meshtags)

    E_elastic, E_kinetic = assemble_scalar(form(Energy_elastic) ), assemble_scalar(form(Energy_kinetic) )    
    metrics = {"time":t,
                     "Energies (elastic)": E_elastic,
                    "Energies (kinetic)": E_kinetic,
                    "Energies (total)": E_elastic+E_kinetic}
    postprocess.log("dict", t, metrics)
    #!SECTION

    # SECTION TIME EVOLUTION
    last_time_measure = process_time()

    while t < T:
        t += dt 
        
        postprocess.log("dict", t, {"Status" : "Solving linear problem", "time" : t, "progress" : (t-t0)/T })

        problem_el.solve()
        
        
        #SECTION - METRICS 1/2: COMPUTE TEST FUNCTION FOR THE RIGHT ORTHOGONALITY MEASURE
        d0_, d1_, d_diff = Function(D), Function(D), Function(D)
        d0_.interpolate(u0.sub(2))
        d1_.interpolate(u.sub(2))
        d_diff.x.array[:] = d1_.x.array[:]- d0_.x.array[:]
        def ind_expr(x: np.ndarray)-> np.ndarray:
            """
            Expression to describe the function:
            $\mathcal{I} ( d^{j-1} \sign{\Tilde{d}^j - d^{j-1}}$
            """
            d0x = eval_continuous_function(d0_,x).T # has shape (#nodes, dim)
            d1x = eval_continuous_function(d1_,x).T # has shape (#nodes, dim)
            res = np.einsum('ij,ik->i', d1x - d0x, d0x) # inner product, has shape (#nodes,)
            res = np.sign( res ) # sign function, has shape (#nodes,)
            res = d0x * res[:,np.newaxis] # scalar vector product,  has shape (#nodes, dim)
            return res.T #has shape (dim, #nodes)  
        test_func_nodal_orth = Function(D)
        test_func_nodal_orth.interpolate(ind_expr)
        nodal_orthogonality_L1 = assemble_scalar(form(inner(d1_ -d0_,test_func_nodal_orth)*dml))
        #!SECTION
        
        # SECTION - NODAL PROJECTION STEP
        if projection:
            """
            The method Pi_h applies the nodal orthogonal projection:
            $ \mathcal{I}_h (\Tilde{d}^j_{o}/\abs{\Tilde{d}^j_{o}}) $
            with 
            $\Tilde{d}^j_{o} := (d^{j-1} + (I-d^{j-1}\otimes d^{j-1}) (\Tilde{d}^j-d^{j-1})) $
            """
            d_projected = Function(D)
            d_projected.interpolate( Pi_h(u.sub(2).collapse(), u0.sub(2).collapse()))
            
            err_projection = errornorm(d_projected, u.sub(2).collapse(), norm = "L2", degree_raise = 0)     
            E_p_diff =  assemble_scalar(form(inner(grad(d_projected),grad(d_projected))*dx - inner(grad(d),grad(d))*dx))

            u.sub(2).interpolate(d_projected) # update result            
        #!SECTION projection step 
        
        
        
        computation_time += process_time() - last_time_measure

        #SECTION - METRICS 2/2: ENERGY AND ERROR MEASURES
        # Energies
        E_elastic, E_kinetic = assemble_scalar(form(Energy_elastic)), assemble_scalar(form(Energy_kinetic) )        
        # Analyticl errors
        errorL2, errorinf = "No exact solution available", "No exact solution available"
        if experiment.has_exact_solution:   
            errorL2 = experiment.compute_error(u.sub(2),t,norm = "L2", degree_raise = 3)   
            errorinf = experiment.compute_error(u.sub(2).collapse(),t,norm = "L2", degree_raise = 0)      
        
        # n = FacetNormal(domain)
        # ds = Measure("ds", domain=domain)
        
        # Collecting metrics
        metrics = {"time":t,
                "Energies (elastic)": E_elastic,
                    "Energies (kinetic)": E_kinetic,
                    "Energies (total)": E_elastic+E_kinetic,
                    "unit norm err": test_unit_norm(u.sub(2).collapse()),  
                    "nodal orthogonality (L1)": nodal_orthogonality_L1,                  
                    "nodal orthogonality (Linfty)": test_ptw_orthogonality(d_diff, d0_),
                    "computation time": computation_time,
                    "errorL2": errorL2,
                    "errorinf": errorinf,
                    "datetime": str(ctime())}
        if projection:
            metrics_P = {"projection error (L2)": err_projection,
                    "Energy increase by projection": E_p_diff,}
        else: metrics_P ={}
        postprocess.log("dict", t, metrics | metrics_P) 
        #!SECTION

       
        
        
        # SECTION - UPDATE AND POSTPROCESSING FUNCTIONS
        u0.x.array[:] = u.x.array[:]

        v_out.interpolate(u0.sub(0))
        p_out.interpolate(u0.sub(1))
        d_out.interpolate(u0.sub(2))
        q_out.interpolate(u0.sub(3))
        postprocess.log_functions(t, {"v": v_out, "p":p_out, "d":d_out, "q":q_out}, mesh = domain, meshtags = meshtags)
        #!SECTION

    #!SECTION TIME EVOLUTION

    postprocess.close()

#!SECTION GENERAL METHOD


# SECTION NUMERICAL SCHEME BINDINGS
def LL2(experiment, args, postprocess=None, solver_metadata = {"ksp_type": "bcgs", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}):
    """
    linear solver without projection and without mass-lumping
    """
    postprocess.log("dict", "static",{"model": "Linear scheme for the Ericksen-Leslie equations", "experiment":experiment.name})
    linear_method(experiment, args, projection=False, use_mass_lumping=False, postprocess = postprocess, solver_metadata = solver_metadata)

def Lh(experiment, args, postprocess=None, solver_metadata = {"ksp_type": "bcgs", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}):
    """
    linear solver without projection and with mass-lumping
    """
    postprocess.log("dict", "static",{"model": "Linear scheme for the Ericksen-Leslie equations, mass-lumping", "experiment":experiment.name})
    linear_method(experiment, args, projection=False, use_mass_lumping=True, postprocess = postprocess, solver_metadata = solver_metadata)

def LL2P(experiment, args, postprocess=None, solver_metadata = {"ksp_type": "bcgs", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}):
    """
    linear solver without projection and without mass-lumping
    """
    postprocess.log("dict", "static",{"model": "Linear scheme for the Ericksen-Leslie equations, with nodal projection", "experiment":experiment.name})
    linear_method(experiment, args, projection=True, use_mass_lumping=False, postprocess = postprocess, solver_metadata = solver_metadata)


def LhP(experiment, args, postprocess=None, solver_metadata = {"ksp_type": "bcgs", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}):
    """
    linear solver without projection and with mass-lumping
    """
    postprocess.log("dict", "static",{"model": "Linear scheme for the Ericksen-Leslie equations, mass-lumping, with nodal projection", "experiment":experiment.name})
    linear_method(experiment, args, projection=True, use_mass_lumping=True, postprocess = postprocess, solver_metadata = solver_metadata)

#!SECTION BINDINGS


