import FFToeplitz as toe
import bempp.api.assembly.discrete_boundary_operator
import bempp.api.assembly.blocked_operator
import numpy as np
#import copy
from bempp.api.assembly.discrete_boundary_operator import _DiscreteOperatorBase
from bempp.api.assembly.blocked_operator import (projections_from_grid_functions_list, grid_function_list_from_coefficients)
from scipy.sparse.linalg import gmres as sp_gmres
from bempp.api.assembly.grid_function import GridFunction

class Toe_zero_data():
    def __init__(self, zeroEls_Dir, zeroEls_Neu):
        self.P0 = np.array(zeroEls_Dir)
        self.P1 = np.array(zeroEls_Neu)

class DiscreteBTTBOperator(_DiscreteOperatorBase):
    """ New discrete operator which is very memory efficient """
    def __init__(self, T):
        self.as_Toe = T
        super().__init__(T.dtype, T.shape)

    def _matvec(self,v):
        return self.as_Toe.matVec(v)

def get_weak_toe_form(dsc_op,zero_obj):
    # if block operator, call recursively over each block:
    if dsc_op.__class__.__name__ == 'BlockedOperator':
        dsc_ops = np.empty((dsc_op.ndims[0], dsc_op.ndims[1]), dtype="O")

        for i in range(dsc_op.ndims[0]):
            for j in range(dsc_op.ndims[1]):
                dsc_ops[i, j] = get_weak_toe_form(dsc_op._operators[i, j],zero_obj)
        return bempp.api.assembly.blocked_operator.BlockedDiscreteOperator(dsc_ops)

    elif dsc_op.__class__.__name__ == '_SumBoundaryOperator':
        return get_weak_toe_form(dsc_op._op1,zero_obj) + get_weak_toe_form(dsc_op._op2,zero_obj)
    elif dsc_op.__class__.__name__ == '_ScaledBoundaryOperator':
        return dsc_op._alpha*get_weak_toe_form(dsc_op._op,zero_obj)
    else:
        # determine spaces, and split in case of p0 space:
        if dsc_op.domain.identifier == 'p1_continuous':
            trial_blocks = [dsc_op.domain.global_dof_count]
            zeros_trial = zero_obj.P1
        elif dsc_op.domain.identifier == 'p0_discontinuous':
            trial_blocks = [int(dsc_op.domain.global_dof_count/2),int(dsc_op.domain.global_dof_count/2)]
            zeros_trial = zero_obj.P0

        if dsc_op.dual_to_range.identifier == 'p1_continuous':
            test_blocks = [dsc_op.dual_to_range.global_dof_count]
            zeros_test = zero_obj.P1
        elif dsc_op.dual_to_range.identifier == 'p0_discontinuous':
            test_blocks = [int(dsc_op.dual_to_range.global_dof_count/2),int(dsc_op.dual_to_range.global_dof_count/2)]
            zeros_test = zero_obj.P0

        # #check for scaling:
        # if dsc_op.__class__.__name__ == '_ScaledBoundaryOperator':
        #     scale = dsc_op._alpha
        #     scaled = True
        # else:
        #     scale = 1
        #     scaled = False

        #construct general discrete opartor, which takes subspaces as arguments
        # if not scaled:
        if dsc_op.descriptor.identifier == 'helmholtz_hypersingular_boundary':
            kwave = dsc_op.descriptor.options[0]
            lam_operator = lambda space: bempp.api.operators.boundary.helmholtz.hypersingular(space, [], space, kwave)
        elif dsc_op.descriptor.identifier == 'helmholtz_single_layer_boundary':
            kwave = dsc_op.descriptor.options[0]
            lam_operator = lambda space: bempp.api.operators.boundary.helmholtz.single_layer(space, [], space, kwave)
        elif dsc_op.descriptor.identifier == 'l2_identity':
            lam_operator = lambda space: bempp.api.operators.boundary.sparse.identity(space, [], space)
            lam_operatorX = lambda space1,space2: bempp.api.operators.boundary.sparse.identity(space1, [], space2)
        else:
            raise ValueError("Haven't coded for the operator:" + dsc_op.descriptor.identifier)
        # else:
        #     if dsc_op._op.descriptor.identifier == 'helmholtz_hypersingular_boundary':
        #         kwave = dsc_op._op.descriptor.options[0]
        #         lam_operator = lambda space: bempp.api.operators.boundary.helmholtz.hypersingular(space, [], space, kwave)
        #     elif dsc_op._op.descriptor.identifier == 'helmholtz_single_layer_boundary':
        #         kwave = dsc_op._op.descriptor.options[0]
        #         lam_operator = lambda space: bempp.api.operators.boundary.helmholtz.single_layer(space, [], space, kwave)
        #     elif dsc_op._op.descriptor.identifier == 'l2_identity':
        #         lam_operator = lambda space: bempp.api.operators.boundary.sparse.identity(space, [], space)
        #         lam_operatorX = lambda space1, space2: bempp.api.operators.boundary.sparse.identity(space1, [], space2)

        def sub_mat_fun(test_start, test_end, trial_start, trial_end):
            inds1 = np.arange(test_start,test_end)
            inds2 = np.arange(trial_start,trial_end)
            space2 = dsc_op.domain
            space1 = dsc_op.dual_to_range
            if space1 == space2:
                if (len(inds1) == 1) & (len(inds2) == 1) & (space1.identifier == 'p0_discontinuous'):
                    #inds1_
                    inds1 = np.append(inds1,inds1+1)
                    inds2 = np.append(inds2,inds2+1)
                    mat_2x2 = get_sub_inds_same_space(inds1,inds2,lam_operator,space1)
                    mat = mat_2x2[0,0]
                else:
                    mat = get_sub_inds_same_space(inds1,inds2,lam_operator,space1)
                    #mat = get_sub_inds_same_spacev2(inds1,inds2,dsc_op)
            else:
                mat = get_sub_inds_different_spaces(inds1,inds2,lam_operatorX,space1,space2)
            if mat.__class__.__name__ == 'csr_matrix':
                return mat.A.ravel()
            else:
                return mat.ravel()

        return construct_discrete_Toe_operator(sub_mat_fun,test_blocks,trial_blocks,zeros_test,zeros_trial)


def construct_discrete_Toe_operator(mat_fun, blockDOFs_test, blockDOFs_trial, zero_els_test, zero_els_trial):
    num_blocks_test = len(blockDOFs_test)
    num_blocks_trial= len(blockDOFs_trial)

    # should we be aiming for blocked or unblocked discrete operators?
    # (rhetorical question, not something to be adressed later)
    if num_blocks_test > 1 or num_blocks_trial>1:
        blocked_output = True
    else:
        blocked_output = False

    #Z = np.empty(3)
    T = np.empty((num_blocks_test,num_blocks_trial), dtype=object)

    mat_sizes_test = np.hstack([0,np.cumsum(blockDOFs_test)])
    mat_sizes_trial = np.hstack([0,np.cumsum(blockDOFs_trial)])

    for n_ in range(num_blocks_test):

        #rearrange the indices of zero padding so that it's relative to this block
        #formerly zL
        zero_test_inds = (mat_sizes_test[n_] <= np.array(zero_els_test)) & (np.array(zero_els_test) < mat_sizes_test[n_+1])
        zero_test_block = zero_els_test[zero_test_inds] - mat_sizes_test[n_]
        zL = zero_test_block
        #Z[n_] = zL ## not sure if this actually gets used - which would be good

        for m_ in range(num_blocks_trial):
            N = int(blockDOFs_test[n_]**.5)
            M = int(blockDOFs_trial[m_]**.5)

            #rearrange the indices of zero padding so that it's relative to this block
            # formerly zR
            zero_trial_inds = ( mat_sizes_trial[m_] <= np.array(zero_els_trial)) & (np.array(zero_els_trial)  < mat_sizes_trial[m_+1]) 
            zero_trial_block = zero_els_trial[zero_trial_inds] - mat_sizes_trial[m_]
            zR = zero_trial_block

            # set their values
            Dd = mat_fun(mat_sizes_test[n_], mat_sizes_test[n_]+1, mat_sizes_trial[m_], mat_sizes_trial[m_]+1)
            Du = mat_fun(mat_sizes_test[n_], mat_sizes_test[n_]+1, (1+mat_sizes_trial[m_]), (M+mat_sizes_trial[m_]))
            Dl = mat_fun((1+mat_sizes_test[n_]),(N+mat_sizes_test[n_]),mat_sizes_trial[m_],mat_sizes_trial[m_]+1)

            # initialise off-diagonal blocks:
        
            Ud = np.zeros(M-1,dtype=np.complex_)
            Uu = np.zeros((M-1,M-1),dtype=np.complex_)
            Ul = np.zeros((M-1,N-1),dtype=np.complex_)
            Ld = np.zeros((N-1),dtype=np.complex_)
            Lu = np.zeros((N-1,M-1),dtype=np.complex_)
            Ll = np.zeros((N-1,N-1),dtype=np.complex_)
            
            outBlock_x = mat_sizes_trial[m_]
            outBlock_y = mat_sizes_test[n_]

            for m in range(1,M):
                Ud[m-1] = mat_fun(outBlock_y, outBlock_y+1, outBlock_x+M*m, outBlock_x+M*m+1)
            for n in range(1,N):
                Ld[n-1] = mat_fun(outBlock_y+N*n, outBlock_y+N*n+1, outBlock_x, outBlock_x+1)
            for n in range(1,N):
                intBlockStart = N*n + 1
                intBlockEnd = N*(n+1)
                Ll[n-1][:(N-1)] = mat_fun((outBlock_y + intBlockStart),(outBlock_y + intBlockEnd),outBlock_x, outBlock_x+1)       
                Lu[n-1][:(M-1)] = mat_fun(outBlock_y+intBlockStart-1,outBlock_y+intBlockStart, (outBlock_x + 1),(outBlock_x + M))
            for m in range(1,M):    
                intBlockStart = M*m + 1
                intBlockEnd = M*(m+1)
                Uu[m-1][:(M-1)] = mat_fun(outBlock_y,outBlock_y+1,(outBlock_x + intBlockStart),(outBlock_x + intBlockEnd))
                Ul[m-1][:(N-1)] = mat_fun((outBlock_y + 1),(outBlock_y + N),(outBlock_x + intBlockStart-1),(outBlock_x + intBlockStart-1)+1)
            
            T[n_][m_] = toe.PadBTTB(Dd, Du, Dl, Ud, Uu, Ul, Ld, Lu, Ll, zR, zL)
        
    if blocked_output:
        ops = np.empty((num_blocks_test,num_blocks_trial), dtype="O")
        for n in range(num_blocks_trial):
            for m in range(num_blocks_test):
                ops[m,n] = DiscreteBTTBOperator(T[m][n])
        return bempp.api.assembly.blocked_operator.BlockedDiscreteOperator(ops)
    else:
        return DiscreteBTTBOperator(T[0][0])
    
def get_mesh_el_inds(inds,space):
    mesh_els_per_global_DOF = len(space.global2local[0])
    # create array which contains indices of the mesh elements, which are in the global indices
    mesh_el_inds = np.array([], dtype = int)
    if len(space.global2local[0]) == 1:
        mesh_el_inds = inds
    else:
        counter = 0
        for n in inds:
            for m in range(mesh_els_per_global_DOF):
                #mesh_el_inds[counter] = space.global2local[n][m][0]
                mesh_el_inds = np.append(mesh_el_inds,space.global2local[n][m][0])
                counter = counter + 1
    mesh_el_inds = np.unique(mesh_el_inds)
    return mesh_el_inds

# determine if two basis elements match, between two different subspaces of the same basis
def same_el(n,space1,m,space2):
    #vertices = space1.grid.vertices
    if len(space1.global2local[0]) == 1: #P0 space
        if (space1.grid.vertices[:,space1.grid.elements[:,n]] == space2.grid.vertices[:,space2.grid.elements[:,m]]).all():
            return True
        else:
            return False
    else:
        for j in range(6):
            if (space1.grid.vertices[:,space1.grid.elements[:,space1.global2local[n][j][0]]] != space2.grid.vertices[:,space2.grid.elements[:,space2.global2local[m][j][0]]]).all():
            #(space1.grid.vertices[:,space1.global2local[n][j][0]] != space2.grid.vertices[:,space2.global2local[m][j][0]]).all():
                return False
        return True

def get_sub_inds_different_spaces(inds1,inds2,operator,space1,space2):
    mesh_el_inds1 = get_mesh_el_inds(inds1,space1)
    mesh_el_inds2 = get_mesh_el_inds(inds2,space2)
    mesh_el_inds = np.sort(np.unique(np.hstack([mesh_el_inds1,mesh_el_inds2])))
    num_nodes = len(mesh_el_inds)
    node_inds = np.empty((3,num_nodes),dtype = int)
    elements = space1.grid.elements
    vertices = space1.grid.vertices
    for n in range(num_nodes):
        node_inds[:,n] = elements[:,mesh_el_inds[n]]
    sub_grid = bempp.api.Grid(vertices, node_inds)
    if space1.identifier == 'p0_discontinuous':
        sub_space1 = bempp.api.function_space(sub_grid, "DP", 0, segments=[0], include_boundary_dofs=True)
    else:
        sub_space1 = bempp.api.function_space(sub_grid, "P", 1, segments=[0], include_boundary_dofs = False)
    if space2.identifier == 'p0_discontinuous':
        sub_space2 = bempp.api.function_space(sub_grid, "DP", 0, segments=[0], include_boundary_dofs=True)
    else:
        sub_space2 = bempp.api.function_space(sub_grid, "P", 1, segments=[0], include_boundary_dofs = False)        
        
    inds1_to_combins = np.empty(max(inds1)+1,dtype=int)
    for n in inds1:
        for m in range(sub_space1.global_dof_count):
            if same_el(n,space1,m,sub_space1):
                inds1_to_combins[n] = m
    inds2_to_combins = np.empty(max(inds2)+1,dtype=int)
    for n in inds2:
        for m in range(sub_space2.global_dof_count):
            if same_el(n,space2,m,sub_space2):
                inds2_to_combins[n] = m
    discrete_op = operator(sub_space1,sub_space2)
    mat = discrete_op.weak_form().A
    #return mat[np.ix_(inds1_to_combins[inds1],inds2_to_combins[inds2])]  
    return mat[np.ix_(inds2_to_combins[inds2],inds1_to_combins[inds1])]  

def get_sub_space(inds,space):
    elements = space.grid.elements
    vertices = space.grid.vertices
    mesh_el_inds = get_mesh_el_inds(inds,space)
    num_nodes = len(mesh_el_inds)
    node_inds = np.empty((3,num_nodes),dtype = int)
    for n in range(num_nodes):
        node_inds[:,n] = elements[:,mesh_el_inds[n]]
    sub_grid = bempp.api.Grid(vertices, node_inds)
    if len(space.global2local[0]) == 1:
        sub_space = bempp.api.function_space(sub_grid, "DP", 0, segments=[0], include_boundary_dofs=True)
    else:
        sub_space = bempp.api.function_space(sub_grid, "P", 1, segments=[0], include_boundary_dofs = False)
    return sub_space

def get_sub_inds_same_space(inds1,inds2,operator,space):
    #space = operator.domain
    combins = np.unique(np.sort(np.hstack([inds1,inds2])))
    inds1_to_combins = np.zeros(max(inds1)+1,dtype=int)
    for n in inds1:
        inds1_to_combins[n] = np.where(combins == n)[0][0]
    inds2_to_combins = np.empty(max(inds2)+1,dtype=int)
    for n in inds2:
        inds2_to_combins[n] = np.where(combins == n)[0][0]
    sub_space = get_sub_space(combins,space)
    discrete_op = operator(sub_space)
    mat = discrete_op.weak_form().A
    return mat[np.ix_(inds1_to_combins[inds1],inds2_to_combins[inds2])]

def gmres(A_full, dual_to_range_sub, domain_sub, rhs_fun, Z,
            tol=1e-5,restart=None,maxiter=None):
    
    A_toe = get_weak_toe_form(A_full,Z)
    
    if isinstance(A_full, bempp.api.assembly.blocked_operator.BlockedOperatorBase):
        blocked = True
        b_vec = projections_from_grid_functions_list(rhs_fun, dual_to_range_sub)
    else:
        blocked = False
        b_vec = rhs_fun.projections(dual_to_range_sub)
    
    # use scipy's gmres to get the coefficients
    x, info = sp_gmres(
    A_toe, b_vec, tol=tol, restart=restart, maxiter=maxiter
    )
    
    if blocked:
        res_fun = grid_function_list_from_coefficients(x.ravel(), domain_sub)
    else:
        res_fun = GridFunction(domain_sub, coefficients=x.ravel())

    return res_fun, info