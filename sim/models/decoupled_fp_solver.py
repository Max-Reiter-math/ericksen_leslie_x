#!/usr/bin/env python
from argparse import Namespace
import numpy as np
from time import process_time
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem import Function, functionspace, form, assemble_scalar, dirichletbc, locate_dofs_topological, locate_dofs_geometrical
from dolfinx.fem.petsc import LinearProblem, assemble_vector
from ufl import div, dx, grad, inner, VectorElement, FiniteElement, MixedElement, TrialFunction, TrialFunctions, TestFunctions, TestFunction, split, Measure, lhs, rhs, FacetNormal
from sim.models.el_forms import *
from sim.common.common_fem_methods import *
from sim.common.common_methods import get_global_dofs

#TODO - Time measurement

#SECTION - GENERAL METHOD
def decoupled_fp_solver(experiment, args, use_mass_lumping: bool = False, a_tol: float = 1E-5, r_tol: float = 1E-4, max_iters: int = 100, postprocess=None, solver_metadata = [{"ksp_type": "bcgs", "pc_type": "jacobi"}, {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}, {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}]):
    # {"ksp_type": "bcgs", "pc_type": "gamg"}
    # {"ksp_type": "bcgs", "pc_type": "jacobi"}
    # {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
    #  solver_metadata = [{"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},{"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}, {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}]
    
    #SECTION - PARAMETERS
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
    #!SECTION

    submodel            = args.submod
    time_step_control   = False # args.tsc

    fp_tol_rel      = r_tol
    fp_tol_abs      = a_tol
    max_fp_iters    = max_iters
    
    #add all local variables for transparency    
    postprocess.log("dict","static",{"model.vars":dict(locals())}, visible =False)

    mesh        = experiment.mesh
    meshtags   = experiment.meshtags

    initial_conditions          = experiment.initial_conditions
    boundary_conditions         = experiment.boundary_conditions

    computation_time  = 0
    last_time_measure = process_time()

    #SECTION - FUNCTION SPACES AND FUNCTIONS  
    P2          = VectorElement("Lagrange", mesh.ufl_cell(), 2, dim = dim)
    P1          = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    me          = MixedElement(P2, P1)
    TH          = functionspace(mesh, me)
    D, Y        = functionspace(mesh, ElementMetaData("Lagrange", 1 , shape=(dim,))), functionspace(mesh, ElementMetaData("Lagrange", 1 , shape=(dim,)))
    TensorF     = functionspace(mesh, ElementMetaData("Lagrange", 1 , shape=(dim, dim)))

    V, mapV = TH.sub(0).collapse()
    Q, mapQ = TH.sub(1).collapse()

    # Functions to use as output tool
    v_out, p_out = Function(V), Function(Q)

    vl1,pl1             = TrialFunctions(TH)
    dl1, ql1            = TrialFunction(D), TrialFunction(Y)
    a,h                 = TestFunctions(TH)
    c, b                = TestFunction(D), TestFunction(Y)
    ### current iterate
    u1                  = Function(TH)
    v1,p1               = split(u1)
    d1, q1              = Function(D), Function(Y)
    ### initial condition / previous iterate
    u0                  = Function(TH)
    v0,p0               = split(u0)
    d0, q0              = Function(D), Function(Y)
    ### Current and previous iterate of the fixpoint iteration
    ul                  = Function(TH)
    vl,pl               = split(ul)
    dl, ql              = Function(D), Function(Y)
    ul0                 = Function(TH)
    vl0,pl0             = split(ul0)
    dl0, ql0            = Function(D), Function(Y)
    ### time average of d corresponding to d^{j-1/2}
    dl1_ = 0.5* ( dl1 + d0)
    dl0_ = 0.5* ( dl0 + d0)
    dl_ = 0.5* ( dl + d0)
    ### projected gradient onto P1
    grad_d0_project = Function(TensorF)
    #!SECTION

    

    # SECTION VARIATONAL FORMULATION
    postprocess.log("dict", "static",{"Status" : "Creating variational formulation"})
    # Defining mass lumping
    dml = Measure("dx", domain = mesh, metadata = {"quadrature_rule": "vertex", "quadrature_degree": 0}) # needed for error computatoin
    if use_mass_lumping:
        dxL = dml
    else:
        dxL = dx

    # Define Energies    
    Energy_kinetic = 0.5*inner(v1, v1) *dx  
    Energy_elastic = 0.5*const_A*inner( grad(0.5* ( d1 + d0)), grad(0.5* ( d1 + d0)))*dx 
    Energy_total = Energy_kinetic + Energy_elastic
  
    # Momentum equation
    def momentum_eq(vl1,  pl1,  dl0_, ql0, a):
        eq =  inner( vl1 - v0  , a )*dx 
        eq += dt*Convection_Velocity_Temam( vl1, v0, a) *dx 
        eq += - dt*inner(pl1,div(a))*dx + dt*div(vl1)*h*dx 
        if use_mass_lumping:
            eq += - dt*v_el* T_E(dl0_, dl0_, grad_d0_project, ql0, a, dim, submodel = submodel)*dxL
        else:
            eq += - dt*v_el* T_E(dl0_, dl0_, grad(d0), ql0, a, dim, submodel = submodel)*dx
        eq += dt*T_D(mu_1, mu_4, mu_5, mu_6, lam, v_el, d0,  vl1, a, dim, submodel = submodel)*dx 
        TL = T_L( lam, dl0_, dl0_, dl0_, ql0, a, dim, submodel = submodel)
        if TL != None: 
            eq += dt*TL*dxL    
        return eq
    Eq1 = momentum_eq(vl1, pl1, dl0_, ql0, a)
    ### consistent form of momentum equation
    cEq1 = momentum_eq(vl, pl,  dl_, ql, a)

    ## compatibility condition / equation for the variational derivative
    def discr_energy_eq(d, q, b):
        return const_A*inner( grad(d), grad(b))*dx - inner(q,b)*dxL
        
    Eq2 = discr_energy_eq(dl0_, ql1, b)  
    #+ eps*inner(dl0_,n)*inner(b,n)*ds #- inner(b, dot( outer(n,n), dot(grad(dl0_), n)))* ds\
    cEq2 = discr_energy_eq( dl_, ql, b) 
    

    ## director equation
    def director_eq( dl1, dl1_, dl0_,  ql0, vl0, c):
        eq = inner(dl1 - d0, c)*dxL
        if use_mass_lumping:
            eq += dt*v_el*T_E(dl0_, dl1_, grad_d0_project, c, vl0, dim, submodel = submodel)*dxL
        else:
            eq += dt*v_el*T_E(dl0_, dl1_, grad(d0), c, vl0, dim, submodel = submodel)*dx
        eq += dt*D_D(dl0_, dl1_,ql0, c, dim, submodel = submodel)*dxL 
        TL = T_L( lam, dl0_, dl1_, dl0_, c, vl0, dim, submodel = submodel)
        if TL != None: eq += - dt*TL*dxL    
        return eq
    Eq3 = director_eq( dl1, dl1_, dl0_,  ql0, vl0, c)
    cEq3 = director_eq( dl, dl_, dl_, ql, vl, c)

    A1 = lhs(Eq1)
    L1 = rhs(Eq1)

    A2 = lhs(Eq2)
    L2 = rhs(Eq2)

    A3 = lhs(Eq3)
    L3 = rhs(Eq3)   
    #!SECTION

    # SECTION INITIAL CONDITIONS
    postprocess.log("dict","static",{"Status" : "Setting initial conditions"})
    u0.sub(0).interpolate(initial_conditions["v"]) 
    u0.sub(1).interpolate(initial_conditions["p"]) 
    d0.interpolate(initial_conditions["d"]) 
    if use_mass_lumping: 
        grad_d0_project.interpolate(project_lumped(grad(d0),TensorF))

    # COMPUTATION OF INITIAL DIVERGENCE
    eq0=discr_energy_eq(d0, ql1, b)  
    problem0 = LinearProblem(lhs(eq0), rhs(eq0),  bcs=[], u=q0, petsc_options=solver_metadata[1])
    problem0.solve()
    # Processing initial conditions before time evolution
    q1.x.array[:] = q0.x.array[:]    
    u1.x.array[:] = u0.x.array[:]
    d1.x.array[:] = d0.x.array[:]
    #!SECTION

    #SECTION - BOUNDARY CONDITIONS
    postprocess.log("dict", "static",{"Status" : "Setting boundary conditions"})
    bcs_v, bcs_d, bcs_q = [], [], []
    for bcwofs in boundary_conditions:
        if bcwofs.type == "Dirichlet":
            if bcwofs.quantity == "v":
                bcwofs.set_fs((TH.sub(0), V))
                bcs_v.append(bcwofs.bc)
            elif bcwofs.quantity == "d":
                bcwofs.set_fs(D)
                bcs_d.append(bcwofs.bc)
                # BOUNDARY CONDITIONS FOR AUXILIARY VARIABLE
                #NOTE - We assume that the initial condition for the director field fulfills the boundary conditions imposed
                q_bc = Function(Y)
                q_bc.interpolate(q0)
                bcs_q = [dirichletbc(q_bc, locate_dofs_geometrical(Y, experiment.boundary))]
        elif bcwofs.type == "Neumann":
            if bcwofs.quantity == "v":
                Eq1 += bcwofs.set_fs(V,a)
            elif bcwofs.quantity == "d":
                Eq2 += bcwofs.set_fs(Y,b)
        # elif bcwofs.type == "Robin":
        else: postprocess.log("dict", "static",{"Warning" : "Boundary conditions of type "+bcwofs.type+" are currently not implemented and will be ignored..."} )

    
    #!SECTION

    #SECTION - SETTING UP LINEAR PROBLEM
    problem1 = LinearProblem(A1, L1,  bcs=bcs_v, u=ul, petsc_options=solver_metadata[0]) 
    problem2 = LinearProblem(A2, L2,  bcs=bcs_q, u=ql, petsc_options=solver_metadata[1])
    problem3 = LinearProblem(A3, L3,  bcs=bcs_d, u=dl, petsc_options=solver_metadata[2])
    #!SECTION
     
    
    
    

    computation_time += process_time() - last_time_measure

    #SECTION - POSTPROCESSING FOR t=0
    v_out.interpolate(u0.sub(0))
    p_out.interpolate(u0.sub(1))
    postprocess.log_functions(t, {"v": v_out, "p": p_out, "d": d0, "q": q0}, mesh = mesh, meshtags = meshtags)

    E_elastic, E_kinetic = assemble_scalar(form(Energy_elastic) ), assemble_scalar(form(Energy_kinetic) )
    metrics =       {"time": t,
                     "Energies (elastic)": E_elastic,
                    "Energies (kinetic)": E_kinetic,
                    "Energies (total)": E_elastic+E_kinetic}
    postprocess.log("dict", t, metrics)
    #!SECTION
    
    #SECTION - TIME EVOLUTION
    E0 = E_elastic + E_kinetic
    dt0 = dt # necessary for time step control

    last_time_measure = process_time()

    while t <= T:
        ### updating time
        if time_step_control: 
            dt = dt0 * (1 + E0) / (1 + E_elastic + E_kinetic)
        t += dt 
        
        fp_err_v, fp_err_d  = np.inf, np.inf
        val_v, val_d = 0, 0
        fp_iter = 0

        #SECTION - FIXPOINT ITERATION
        while not (fp_err_v <= np.maximum(fp_tol_abs, fp_tol_rel * val_v ) and fp_err_d<= np.maximum(fp_tol_abs, fp_tol_rel * val_d )) and (fp_iter < max_fp_iters):
            fp_iter += 1
            last_time_measure = process_time()
            postprocess.log("dict", t, {"Status" : "Solving linear problem","fp iteration":fp_iter, "time" : t, "progress" : (t-t0)/T })

            postprocess.log("dict", t,{"Status" : "Solving momentum eq", "time": t}) 
            problem1.solve()
            
            postprocess.log("dict", t,{"Status" : "Solving director eq", "time": t}) 
            problem3.solve()     
            
            postprocess.log("dict", t,{"Status" : "Solving energy eq", "time": t})
            problem2.solve()            

            #SECTION - METRICS WITHIN FP ITERATION
            ## fp error computation
            elv = vl - vl0
            eld = dl - dl0
            elq = ql - ql0
            # FP ERROR AS ABSOLUTE OR RELATIVE??
            fp_err_v = np.sqrt(assemble_scalar(form( 
                inner(elv, elv) *dx)))
            val_v    = np.sqrt(assemble_scalar(form( 
                inner(vl0, vl0) *dx)))
            fp_err_d = np.sqrt(assemble_scalar(form(inner(eld, eld) *dx  
                +  inner(grad(eld),grad(eld))*dx
                ))) 
            val_d    = np.sqrt(assemble_scalar(form(inner(dl0, dl0) *dx  
                +  inner(grad(dl0),grad(dl0))*dx
                ))) 
            fp_err_q = np.sqrt(assemble_scalar(form(inner(elq, elq) *dx)))
            fp_err_abs = fp_err_v + fp_err_d
            computation_time += process_time() - last_time_measure

            
            consistency_err = {
                    1: np.linalg.norm(assemble_vector(form(cEq1))[:], ord =np.inf), 
                    2: np.linalg.norm(assemble_vector(form(cEq2))[:], ord =np.inf), 
                    3: np.linalg.norm(assemble_vector(form(cEq3))[:], ord =np.inf)
                }
            E_elastic, E_kinetic = assemble_scalar(form(Energy_elastic)), assemble_scalar(form(Energy_kinetic) )
            
            error = "N/A"
            if experiment.has_exact_solution:     
                error = experiment.compute_error(dl,t, degree_raise = 3)           

            metrics = {"time": t,
                        "fp err abs": fp_err_abs,
                        "el_v (L2)": fp_err_v,
                        "el_d (H1)": fp_err_d,
                        "el_q (L2)": fp_err_q,
                        "L2(vl0)": val_v,
                        "H1(dl0)": val_d,
                        "Energies (elastic)": E_elastic,
                        "Energies (kinetic)": E_kinetic,
                        "Energies (total)": E_elastic+E_kinetic,
                        "consistency err": consistency_err,
                        "unit norm err": test_unit_norm(d1),
                        "computation time": computation_time,
                        "fp iteration": fp_iter,
                        "error": error}
            postprocess.log("dict", t, metrics) 
            #!SECTION

            #SECTION - UPDATE WITHIN FP ITERATION
            ul0.x.array[:] = ul.x.array[:]
            dl0.x.array[:] = dl.x.array[:]
            ql0.x.array[:] = ql.x.array[:]
            #!SECTION

        #!SECTION Fixed point iteration
        postprocess.log("dict", t, metrics) 
        # Stop if tolerance is not achieved
        if not (fp_err_v <= np.maximum(fp_tol_abs, fp_tol_rel * val_v ) and fp_err_d<= np.maximum(fp_tol_abs, fp_tol_rel * val_d )): raise RuntimeError("Fixpoint solver did not converge.")
        
        #SECTION - UPDATE AND METRICS COMPUTATION AFTER TIME EVOLUTION STEP
        u1.x.array[:] = ul.x.array[:]
        d1.x.array[:] = dl.x.array[:]
        q1.x.array[:] = ql.x.array[:]

        # Potential metrics computation here

        #SECTION - UPDATE AND SAVING AFTER TIME EVOLUTION STEP
        u0.x.array[:] = u1.x.array[:]
        d0.x.array[:] = d1.x.array[:]
        q0.x.array[:] = q1.x.array[:]
        if use_mass_lumping: 
            grad_d0_project.interpolate(project_lumped(grad(d0),TensorF))
        # saving
        v_out.interpolate(u0.sub(0))
        p_out.interpolate(u0.sub(1))
        postprocess.log_functions(t, {"v": v_out, "p": p_out, "d": d0, "q": q0})

        
        
    #!SECTION TIME EVOLUTION
        
    postprocess.close()

#!SECTION GENERAL METHOD


# SECTION NUMERICAL SCHEME BINDINGS
def FPhD(experiment, args, postprocess=None, 
                        solver_metadata = [{"ksp_type": "bcgs", "pc_type": "jacobi"}, {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}, {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}]):
    """
    FP solver with
        mass-lumping
    """
    postprocess.log("dict", "static",{"model": "Decoupled fixpoint solver with mass-lumping", "experiment":experiment.name})
    decoupled_fp_solver(experiment, args, use_mass_lumping = True, a_tol = 1E-5, r_tol = 1E-4, max_iters = 100, postprocess=postprocess, solver_metadata = solver_metadata)

def FPL2D(experiment, args, postprocess=None, 
                        solver_metadata = [{"ksp_type": "bcgs", "pc_type": "jacobi"}, {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}, {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}]):
    """
    FP solver without mass-lumping
    """
    postprocess.log("dict", "static",{"model": "Decoupled fixpoint solver", "experiment":experiment.name})
    decoupled_fp_solver(experiment, args, use_mass_lumping = False, a_tol = 1E-5, r_tol = 1E-4, max_iters = 100, postprocess=postprocess, solver_metadata = solver_metadata)
#!SECTION