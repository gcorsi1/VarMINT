from dolfin import *

def calc_force_coeffs(u_h,p_h,mu,n,ds,resdict,len_scale=1.0):
    """
    assume that resdict is a defaultdict(list)
    """
    n1 = -n
    u_t = inner(as_vector((n1[1], -n1[0])), u_h)
    drag = assemble(2/len_scale*(mu*inner(grad(u_t), n1)*n1[1] - p_h*n1[0])*ds)
    lift = assemble(-2/len_scale*(mu*inner(grad(u_t), n1)*n1[0] + p_h*n1[1])*ds)
    resdict['c_ds'].append(drag)
    resdict['c_ls'].append(lift)
    return
