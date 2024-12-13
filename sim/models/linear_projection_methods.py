from time import process_time, ctime
from functools import partial
from dolfinx.fem import Function, functionspace, dirichletbc, form, assemble_scalar, dirichletbc, locate_dofs_topological, locate_dofs_geometrical, ElementMetaData, Expression
from dolfinx.fem.petsc import LinearProblem, assemble_vector
from ufl import div, dx, grad, inner, VectorElement, FiniteElement, MixedElement, TrialFunctions, TrialFunction, TestFunction, split, Measure, lhs, rhs, FacetNormal, sqrt, dP, sign, exp, as_ufl
from sim.models.el_forms import *
from sim.common.common_fem_methods import *
from sim.common.meta_bcs import *
from sim.common.error_computation import errornorm

# dolfinx v0.7

#SECTION - GENERAL METHOD
def linear_method(experiment, args, projection: bool, use_mass_lumping: bool, postprocess=None, solver_metadata = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}):
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
    
    d0_, d1_, d_diff = Function(D), Function(D), Function(D) # used later to compute the nodal/cell-wise 

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
        grad_d0 = grad_d0_project
    else:
        dxL = dx
        grad_d0 = grad(d0)

    # SECTION DEFINITION OF ENERGY
    Energy_kinetic = 0.5*inner(v, v) *dx  
    Energy_elastic = 0.5*inner( grad(d ), grad(d ))*dx 
    Energy_total = Energy_kinetic + const_A*Energy_elastic

    def compute_energies(appendix = ""):        
        E_ela   = assemble_scalar(form(Energy_elastic) )
        E_kin   = assemble_scalar(form(Energy_kinetic) )
        E_tot   = E_kin + const_A*E_ela
        metrics = { "Energies (total)"+appendix: E_tot,
                    "Energies (elastic)"+appendix: E_ela,
                    "Energies (kinetic)"+appendix: E_kin
                    }
        return metrics    
    #!SECTION
    
    # SECTION EQUATION SYSTEM DEFINITION

    def momentum_eq(v1,  p1, d0, q1, a):
        # discrete time derivative
        eq =  inner( (v1-v0) , a )*dx 

        # convection term
        eq += dt* ( inner(dot(v0, nabla_grad(v1)), a) + 0.5*div(v0)*inner(v1, a)  )*dx 

        # pressure term and divergence zero condition
        eq += - dt*inner(p1,div(a))*dx + div(v1)*h*dx 

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
            eq += -dt*const_A*v_el*inner( cross(d0, dot(grad_d0 , a)),  cross(d0, q1))*dxL
        else:
            eq += -dt*const_A*v_el*inner( dot( I_dd(d0, d0, dim) , dot(grad_d0 , a)),  q1)*dxL

        # Leslie stress tensor terms
        if submodel == 2:
            if dim ==3:
                eq +=  -dt*lam*inner(cross( d0 , q1), cross(d0, dot(grad_sym(a), d0) ) )*dxL
                eq += dt*inner(cross( d0 ,dot(grad_skw(a),d0)), cross(d0, q1))*dx
            else:
                eq +=  -dt*lam*inner(dot( I_dd(d0,d0, dim) , q1), dot(grad_sym(a), d0))*dxL
                eq += dt*inner(dot( I_dd(d0,d0, dim) ,dot(grad_skw(a),d0)), q1)*dxL

        return eq
        
    def discr_energy_eq(d, q, b):
        # equation for the variational derivative, here discrete divergence (q = \Delta_h d)
        return inner( grad(d), grad(b))*dx - inner(q,b)*dxL 

    def director_eq( d1, d0, q1, v1, c):
        # discrete time derivative
        eq = inner(d1 - d0, c)*dxL

        # dissipative terms
        if dim == 3: 
            eq += dt*const_A*inner( cross(d0, q1), cross(d0, c))*dxL
        else: 
            eq += dt*const_A*inner(q1 , dot( I_dd(d0,d0, dim) , c) )*dxL

        # Ericksen stress tensor terms
        if dim ==3:
            eq += dt*v_el*inner( cross(d0, dot(grad_d0 , v1)),  cross(d0, c))*dxL
        else:
            eq += dt*v_el*inner( dot( I_dd(d0, d0, dim) , dot(grad_d0 , v1)),  c)*dxL
   
        # Leslie stress tensor terms
        if submodel == 2:
            if dim ==3:
                eq +=  dt*lam*inner(cross( d0 , c), cross(d0, dot(grad_sym(v1), d0) ) )*dxL
                eq += - dt*inner(cross( d0 ,dot(grad_skw(v1),d0)), cross(d0, c))*dxL
            else:
                eq +=  dt*lam*inner(dot( I_dd(d0,d0, dim) , c), dot(grad_sym(v1), d0))*dxL
                eq += - dt*inner(dot( I_dd(d0,d0, dim) ,dot(grad_skw(v1),c)), d0)*dxL
  
        return eq
    
    #!SECTION
    
    Eq1 = momentum_eq(v1, p1, d0, q1, a)
    Eq2 = discr_energy_eq(d1, q1, b)  
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
                bcq = meta_dirichletbc("q", bcwofs.find_dofs, q_ic, marker = bcwofs.marker , entity_dim = bcwofs.dim, entities = bcwofs.entities)
                bcq.set_fs((F.sub(3), Y))
                bcs.append(bcq.bc)
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

    metrics = {"time":t}
    metrics.update(compute_energies())
    postprocess.log("dict", t, metrics)
    #!SECTION

    # SECTION TIME EVOLUTION
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
            #SECTION - METRICS BEFORE PROJECTION STEP
            metrics.update(compute_energies(appendix=" b4 projection"))
            d0_.interpolate(u0.sub(2))
            d1_.interpolate(u.sub(2))
            d_diff.x.array[:] = d1_.x.array[:]- d0_.x.array[:]
            
            test_func_nodal_orth = Function(D)
            test_func_nodal_orth.interpolate(partial(ind_expr, d0_, d1_))
            nodal_orthogonality_L1 = assemble_scalar(form(inner(d1_ -d0_,test_func_nodal_orth)*dml))
            tmp = {
                    "unit norm err b4 projection": test_unit_norm(u.sub(2).collapse()),    
                    "nodal orthogonality (L1) b4 projection": nodal_orthogonality_L1,               
                    "nodal orthogonality (Linfty) b4 projection": test_ptw_orthogonality(d_diff, d0_)}
            metrics.update(tmp)
            #!SECTION

            # ACTUAL PROJECTION STEP
            d_projected = Function(D)
            d_projected.interpolate( Pi_h(u.sub(2).collapse(), u0.sub(2).collapse()))
            u.sub(2).interpolate(d_projected) # update result            
        #!SECTION projection step 
        
        
        
        computation_time += process_time() - last_time_measure

        #SECTION - METRICS AFTER PROJECTION STEP    
        # Analyticl errors
        errorL2, errorinf = "No exact solution available", "No exact solution available"
        if experiment.has_exact_solution:   
            errorL2 = experiment.compute_error(u.sub(2),t,norm = "L2", degree_raise = 3)   
            errorinf = experiment.compute_error(u.sub(2).collapse(),t,norm = "inf", degree_raise = 0)      
        
        # Collecting metrics
        metrics.update(compute_energies())
        d0_.interpolate(u0.sub(2))
        d1_.interpolate(u.sub(2))
        d_diff.x.array[:] = d1_.x.array[:]- d0_.x.array[:]
        test_func_nodal_orth = Function(D)
        test_func_nodal_orth.interpolate(partial(ind_expr, d0_, d1_))
        nodal_orthogonality_L1 = assemble_scalar(form(inner(d1_ -d0_,test_func_nodal_orth)*dml))
        tmp = {
                    "unit norm err": test_unit_norm(u.sub(2).collapse()),    
                    "nodal orthogonality (L1)": nodal_orthogonality_L1,               
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
        if use_mass_lumping: 
            grad_d0_project.interpolate(project_lumped(grad(d0),TensorF))

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
def LL2(experiment, args, postprocess=None):
    """
    linear solver without projection and without mass-lumping
    """
    postprocess.log("dict", "static",{"model": "Linear scheme for the Ericksen-Leslie equations", "experiment":experiment.name})
    linear_method(experiment, args, projection=False, use_mass_lumping=False, postprocess = postprocess)

def Lh(experiment, args, postprocess=None):
    """
    linear solver without projection and with mass-lumping
    """
    postprocess.log("dict", "static",{"model": "Linear scheme for the Ericksen-Leslie equations, mass-lumping", "experiment":experiment.name})
    linear_method(experiment, args, projection=False, use_mass_lumping=True, postprocess = postprocess)

def LL2P(experiment, args, postprocess=None ):
    """
    linear solver without projection and without mass-lumping
    """
    postprocess.log("dict", "static",{"model": "Linear scheme for the Ericksen-Leslie equations, with nodal projection", "experiment":experiment.name})
    linear_method(experiment, args, projection=True, use_mass_lumping=False, postprocess = postprocess)


def LhP(experiment, args, postprocess=None):
    """
    linear solver without projection and with mass-lumping
    """
    postprocess.log("dict", "static",{"model": "Linear scheme for the Ericksen-Leslie equations, mass-lumping, with nodal projection", "experiment":experiment.name})
    linear_method(experiment, args, projection=True, use_mass_lumping=True, postprocess = postprocess)

#!SECTION BINDINGS

#SECTION - Helper function
def ind_expr(d0_, d1_, x: np.ndarray)-> np.ndarray:
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
#!SECTION

