import numpy as np
from ufl import div, grad, inner, nabla_grad, nabla_div, outer, as_matrix, as_vector, dot, cross, Constant
from petsc4py import PETSc
"""
Advanced UFL operators 
"""

# STANDARD FORMS FOR SIMPLIFICATION
def grad_sym(v):
    """
    symmetric part of the gradient
    """
    return   0.5*(grad(v) + grad(v).T)
def grad_skw(v):
    """
    skew-symmetric part of the gradient
    """
    return   0.5*(grad(v) - grad(v).T)

def a_times_b_times_c(a, b, c):
    """
    a x (b x c) =  (a . c) b - (a . b) c 
    """
    return inner(a,c)*b - inner(a,b)*c 

# def a_times_b(dim, a, b):


"""
old
"""

def Convection_Velocity_Temam( v1, v0, v_test):
    """
    modification of the discrete convection term in order to fulfill skew-symmetry in the 2nd and 3rd component
    """
    return inner(dot(v0, nabla_grad(v1)), v_test) + 0.5*div(v0)*inner(v1, v_test)

# GENERAL FORMS FOR THE DIRECTOR
def I_dd(d1,d2, dim):
    I = as_matrix( np.identity(dim) ) # as_matrix is a UFL function needed in order to identify the var. form. correctly
    M = inner(d1,d2)*I - outer(d1,d2)
    # M=I - outer(d1,d2)
    return M

# THE FOLLOWING METHODS AUTOMATICALLY CHOOSE THE RIGHT SUBMODEL
def T_E(d1, d2, grad_d, q_, v_, dim, submodel = 1):
    if submodel == 1:
        return T_E_1(d1, d2, grad_d, q_, v_, dim)
    elif submodel == 2:
        return T_E_2(d1, d2, grad_d, q_, v_, dim)
    elif submodel == 3:
        return T_E_3(d1, d2, grad_d, q_, v_, dim)
    else:
        raise ValueError("Submodel with index "+str(submodel)+" does not exist.")

def T_D(mu_1, mu_4, mu_5, mu_6, lam, v_el, d,  v, a, dim, submodel = 1):
    if submodel == 1:
        return T_D_1(mu_1, mu_4, mu_5, mu_6, lam, v_el, d,  v, a, dim)
    elif submodel == 2:
        return T_D_2(mu_1, mu_4, mu_5, mu_6, lam, v_el, d,  v, a, dim)
    elif submodel == 3:
        return T_D_3(mu_1, mu_4, mu_5, mu_6, lam, v_el, d,  v, a, dim)
    else:
        raise ValueError("Submodel with index "+str(submodel)+" does not exist.")
    
def T_L( lam, d1, d2, d_, q_, v_, dim, submodel = 1):
    if submodel == 1:
        return T_L_1( lam, d1, d2, d_, q_, v_, dim)
    if submodel == 2:
        return T_L_2( lam, d1, d2, d_, q_, v_, dim)
    if submodel == 3:
        return T_L_3( lam, d1, d2, d_, q_, v_, dim)
    else:
        raise ValueError("Submodel with index "+str(submodel)+" does not exist.")

def D_D(d0, d1, q, q_, dim,  submodel = 1): 
    if submodel == 1:
        return D_D_1(d0, d1, q, q_, dim)
    if submodel == 2:
        return D_D_2(d0, d1, q, q_, dim)
    if submodel == 3:
        return D_D_3(d0, d1, q, q_, dim)
    else:
        raise ValueError("Submodel with index "+str(submodel)+" does not exist.")


# ---
# SUBMODEL 1: FULL SYSTEM
# ---
def T_E_1(d1, d2, grad_d, q_, v_, dim):
    if dim ==3:
        return inner( cross(d1, dot(grad_d , v_)),  cross(d2,q_))
    else:
        return inner( dot( I_dd(d1, d2, dim) , dot(grad_d , v_)),  q_)        

# dissipative parts of the Leslie stress tensor
def T_D_1(mu_1, mu_4, mu_5, mu_6, lam, v_el, d,  v, a, dim):
    form = v_el*(mu_1+lam**2)*inner(inner(d,dot(grad_sym(v),d)),inner(d,dot(grad_sym(a),d)))\
        + (mu_4)*inner( grad_sym(v), grad_sym(a)) \
        + v_el* (mu_5+mu_6-lam**2)*inner( dot(grad_sym(v),d), dot(grad_sym(a),d))
    return form

# rest of the Leslie stress tensor
def T_L_1( lam, d1, d2, d_, q_, v_, dim):
    if dim ==3:
        form = - lam*inner(cross( d1 , q_), cross(d2, dot(grad_sym(v_), d_) ) ) \
                - inner(cross( d1 ,dot(grad_skw(v_),d_)), cross(d2, q_))
    else:
        form = - lam*inner(dot( I_dd(d1,d2, dim) , q_), dot(grad_sym(v_), d_)) \
            - inner(dot( I_dd(d1,d2, dim) ,dot(grad_skw(v_),q_)), d_)
    return form

# Dissipation term of the director equation
def D_D_1(d0, d1, q, q_, dim):    
    if dim == 3: 
        form = inner( cross(d0, q), cross(d1,q_))
    else: 
        form = inner(q , dot( I_dd(d0,d1, dim) , q_) )
    return form

# ---
# SUBMODEL 2: SIMPLIFIED SYSTEM WITH A TRIVIAL LESLIE STRESS TENSOR
# ---
def T_E_2(d1, d2, grad_d, q_, v_, dim):
    if dim ==3:
        return inner( cross(d1, dot(grad_d , v_)),  cross(d2,q_))
    else:
        return inner( dot( I_dd(d1, d2, dim) , dot(grad_d , v_)),  q_)  

def T_D_2(mu_1, mu_4, mu_5, mu_6, lam, v_el, d,  v, a, dim):
    # nu = v_el*(mu_1+lam**2) + mu_4 + v_el* (mu_5+mu_6-lam**2)
    nu = mu_4
    form = nu*inner( grad(v), grad(a))
    return form


# rest of the Leslie stress tensor
def T_L_2( lam, d1, d2, d_, q_, v_, dim):
    return None

# Dissipation term of the director equation
def D_D_2(d0, d1, q, q_, dim):    
    if dim == 3: 
        form = inner( cross(d0, q), cross(d1,q_))
    else: 
        form = inner(q , dot( I_dd(d0,d1, dim) , q_) )
    return form

# ---
# SUBMODEL 2: SIMPLIFIED SYSTEM WITH A TRIVIAL LESLIE STRESS TENSOR AND WITHOUT PROJECTION ONTO THE TANGENTIAL SPACE OF THE DIRECTOR
# ---
def T_E_3(d1, d2, grad_d, q_, v_, dim):
    return inner( dot(grad_d , v_),  q_)

def T_D_3(mu_1, mu_4, mu_5, mu_6, lam, v_el, d,  v, a, dim):
    # nu = v_el*(mu_1+lam**2) + mu_4 + v_el* (mu_5+mu_6-lam**2)
    nu = mu_4
    return nu*inner( grad(v), grad(a))

# rest of the Leslie stress tensor
def T_L_3( lam, d1, d2, d_, q_, v_, dim):
    return None

# Dissipation term of the director equation
def D_D_3(d0, d1, q, q_, dim):    
    return inner(q, q_)