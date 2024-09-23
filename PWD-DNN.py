import torch
import numpy as np
import math
import ot
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def pwdDNN_test(data1, data2, kseq, rhoseq, hyper_params, each_projDim=True, reps=10, train_ratio=0.5, alpha=0.05):

    n1, sample_dim = data1.size()
    n2 = data2.size(0)
    n11 = int(n1*train_ratio)
    n21 = int(n2*train_ratio)
    n12 = n1 - n11
    n22 = n2 - n21
    loc11 = np.random.choice(n1, n11, replace=False)
    loc12 = np.delete(np.arange(n1), loc11)
    data11 = data1[loc11, :]
    data12 = data1[loc12, :]
    loc21 = np.random.choice(n2, n21, replace=False)
    loc22 = np.delete(np.arange(n2), loc21)
    data21 = data2[loc21, :]
    data22 = data2[loc22, :]   

    result = tuningFree(data11, data21, data12, data22, kseq, rhoseq, hyper_params, reps=reps)
    val1 = result['val1']  
    val2 = result['val2']
    result_pwd = getTuningFreeResult(val1, val2, each_projDim=each_projDim, alpha=alpha)
    Tn = result_pwd['statistic']
    Tn = Tn.numpy()
    klen = kseq.size(0)
    rholen = rhoseq.size(0)
    if each_projDim:
        test_thresh = np.concatenate((getQuantile_2side(1-alpha, rholen).repeat(klen), getQuantile_2side(1-alpha, rholen*klen)), axis=None)
    else:
        test_thresh = getQuantile_2side(1-alpha, rholen*klen)
    decision = (Tn > test_thresh)
    
    return {'statistic': Tn, 'decision': decision, 'each_projDim', each_projDim, 'significance level': alpha, \
            'train ratio': train_ratio, 'candidate dim': kseq, 'candidate sparsity': rhoseq}


# @description obtain the (two-sided) test statistic
# @param vals1, vals2: estimated values of discriminative functions for two samples
# @return a dictionary with
#           \key{statistic}: two-sided test statistic
#           \key{scaled distance}: the distance scaled by the precision matrix
def getTuningFreeResult(vals1, vals2, each_projDim=True, alpha=0.05):

    vals1 = vals1.to(device)
    vals2 = vals2.to(device)
    n1, m, l = vals1.size()
    n2 = vals2.size(0)
    if each_projDim:
        Tn2 = torch.zeros(l+1)
        for i in range(l):
            # get the result for a given projected dimension
            cov1 = myCov(vals1[:,:,i]) 
            cov2 = myCov(vals2[:,:,i])
            cov_est = cov1*n2/(n1+n2) + cov2*n1/(n1+n2)
            mu1 = torch.mean(vals1[:,:,i], 0)
            mu2 = torch.mean(vals2[:,:,i], 0)
            wd = math.sqrt(n1*n2/(n1+n2))*(mu2 - mu1) 
            
            precision_matrix = torch.linalg.pinv(cov_est) 
            # precision_matrix = torch.linalg.inv(cov_est) 
            eval, u = torch.linalg.eig(precision_matrix)
            eval = torch.real(eval) # real part
            u = torch.real(u)
            evals = torch.diag(eval**0.5)
            root_precision = torch.matmul(torch.matmul(u, evals), u.t())

            Tn2[i] = torch.max(torch.abs(torch.matmul(root_precision, wd))).to('cpu')          
    else:
        Tn2 = torch.zeros(1)

    if each_projDim:
        index = l
    else:
        index = 0
    # get the result over all candidate parameters
    vals1 = vals1.reshape(n1, m*l)
    vals2 = vals2.reshape(n2, m*l)
    cov1 = myCov(vals1) 
    cov2 = myCov(vals2)
    cov_est = cov1*n2/(n1+n2) + cov2*n1/(n1+n2)
    mu1 = torch.mean(vals1, 0)
    mu2 = torch.mean(vals2, 0)
    wd = math.sqrt(n1*n2/(n1+n2))*(mu2 - mu1)

    precision_matrix = torch.linalg.pinv(cov_est) 
    # precision_matrix = torch.linalg.inv(cov_est) 
    eval, u = torch.linalg.eig(precision_matrix)
    eval = torch.real(eval) # real part
    u = torch.real(u)
    evals = torch.diag(eval**0.5)
    root_precision = torch.matmul(torch.matmul(u, evals), u.t())
    
    Tn2[index] = torch.max(torch.abs(torch.matmul(root_precision, wd))).to('cpu') 
    distance_scaled = torch.matmul(root_precision, wd).to('cpu')

    return {'statistic': Tn2, 'scaled distance': distance_scaled}


# @description get estimation results under given tuning parameters
# @param kseq, a candidate set of projection dimensions
# @param rhoseq, a candidate set of sparsity parameters
# @param hyper_params, parameters used during neural network optimization
# @param reps, the number of different initial directions used; default value: 10
# @return a dictionary with
#           \key{}
# @details 
def tuningFree(data11, data21, data12, data22, kseq, rhoseq, hyper_params, reps=10):

    n11, sample_dim = data11.size()
    n21 = data21.size(0)
    n12 = data12.size(0)
    n22 = data22.size(0)
    rholen = rhoseq.size(0)
    klen = kseq.size(0)
    kmax = torch.max(kseq)
    vals1 = torch.zeros(n12, rholen, klen, device=device)
    vals2 = torch.zeros(n22, rholen, klen, device=device)
    Vs_opt = torch.zeros(sample_dim, kmax, rholen, klen, device=device)
    Tns = torch.zeros(rholen, klen)
    for j in range(klen):
        k = kseq[j]
        hyper_params['in_features'] = k
        hyper_params['node'] = hyper_params['nodeSeq'][j]
        for i in range(rholen):
            rho = rhoseq[i]
            pwd_max = -1
            for i_rep in range(reps):
                V0 = torch.randn(sample_dim, k, device=device)
                V0, _ = torch.linalg.qr(V0)
                p1 = torch.ones(n11, device=device) / n11
                p2 = torch.ones(n21, device=device) / n21
                V = get_projection(data11, data21, V0, p1, p2, rho)                
                dist_mat = pdist(data11, data21, V)
                # pwd = ot.emd2(p1, p2, dist_mat)
                pwd = torch.sum(ot.emd(p1, p2, dist_mat) * dist_mat)
                if pwd >= pwd_max:
                    V_opt = V.clone()
                    pwd_max = pwd.clone()
            Vs_opt[:, 0:k, i, j] = V_opt
            proj_data12 = torch.matmul(data12, V_opt)
            proj_data22 = torch.matmul(data22, V_opt)
            proj_data11 = torch.matmul(data11, V_opt)
            proj_data21 = torch.matmul(data21, V_opt)
            result_pwd = train_Lipschitz_NN(proj_data11, proj_data21, proj_data12, proj_data22, hyper_params)
            vals1[:, i, j] = result_pwd['val1'].squeeze()
            vals2[:, i, j] = result_pwd['val2'].squeeze()
            Tns[i, j] = result_pwd['test statistic']  
            print('projection dimension {}, sparsity parameter {}, corresponding statistic {}'.format(k, rho, Tns[i,j]))  
    Tns = Tns.reshape(rholen*klen,1).squeeze()
    result = {'val1': vals1, 'val2': vals2, 'projection': Vs_opt, 'test statistic': Tns }        
    return result


# @description Obtain the optimal projection (V)
# @param p1, p2: two marginal distributions in the optimal transport problem
# @param old_V: initial value of projection
# @param rho: sparsity parameter
# @detail Proximal gradient method for manifold optimization
#         the optimization problem is the optimal transport problem with a sparsity penalty
#         alternating optimization with respect to V and T_coupling
def get_projection(X, Y, old_V, p1, p2, rho, thresh=1e-4, max_iter=100):

    X = X.to(device)
    Y = Y.to(device)
    old_V = old_V.to(device)
    iter_id = 0
    error = 1
    while error > thresh and iter_id <= max_iter:
        T_coupling = update_OT(X, Y, old_V, p1, p2)
        V = update_projection(X, Y, T_coupling, old_V, rho) 
        error = torch.linalg.norm(V-old_V) / (1+torch.linalg.norm(old_V))
        old_V = V.clone()
        iter_id = iter_id+1
    return V


# @description define the structure of neural nets
class wdnet(nn.Module):
    def __init__(self, hyper_params):
        super().__init__() 
        self.in_features = hyper_params['in_features']
        self.node = hyper_params['node']
        self.hidden_layers = hyper_params['hidden_layers']
        net_list = []
        net_list.append(nn.utils.spectral_norm(nn.Linear(self.in_features, self.node)))
        for i in range(self.hidden_layers-1):
            net_list.append(nn.utils.spectral_norm(nn.Linear(self.node, self.node)))        
        net_list.append(nn.utils.spectral_norm(nn.Linear(self.node, 1)))
        self.layers = nn.ModuleList(net_list)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layers[0](x))
        for i in range(self.hidden_layers-1):
            x = self.relu(self.layers[i+1](x))
        x = self.layers[-1](x)
        return x

# @description train the neural network given projected data
# @param hyper_params: parameters for neural networks
def train_Lipschitz_NN(data1, data2, eval_data1, eval_data2, hyper_params):

    learn_rate = hyper_params['learn_rate']
    nepoch = hyper_params['nepoch']
    n1 = data1.size(0)
    n2 = data2.size(0)
    mywdnet = wdnet(hyper_params)
    mywdnet.to(device)
    optimizer = optim.Adam(mywdnet.parameters(), lr=learn_rate, betas=(0.5, 0.999))
    mywdnet.train()
    for epoch in range(nepoch):
        out1 = mywdnet(data1)
        out2 = mywdnet(data2)
        error = out1.mean() - out2.mean()
        optimizer.zero_grad()
        error.backward()
        optimizer.step()
    mywdnet.eval()
    with torch.no_grad():
        val1 = mywdnet(eval_data1)
        val2 = mywdnet(eval_data2)
        wasser_distance = val2.mean() - val1.mean()
        n1 = eval_data1.size(0)
        n2 = eval_data2.size(0)
        myvar = torch.var(val2, unbiased=False)/n2 + torch.var(val1, unbiased=False)/n1        
        Tn = wasser_distance / torch.sqrt(myvar)
        Tn = Tn.to('cpu')
        wasser_distance = wasser_distance.to('cpu')
        result = {'test statistic': Tn, 'val1': val1, 'val2': val2, 'dist': wasser_distance}
    return result

# @description update the projection (V) with the proximal gradient method for manifold optimization
# @detail given the transport plan (T_coupling) and the projection (old_V) from the last iteration
def update_projection(X, Y, T_coupling, old_V, rho):

    n1 = X.size(0)
    n2 = Y.size(0)
    k = old_V.size(1)
    dist_mat = pdist(X, Y, old_V)
    obj_old = -torch.sum(T_coupling*dist_mat) + rho*torch.linalg.norm(old_V, ord=1)

    dist_mat = torch.sqrt(dist_mat**2 + 1e-6)
    PP = T_coupling/dist_mat
    G = 0
    for i in range(n1):
        x_temp = X[i,:].repeat(n2, 1).T #p*n2
        PP_temp = torch.diag(PP[i,:])
        G = G + torch.matmul(torch.matmul(x_temp-Y.T, PP_temp), (x_temp-Y.T).T)
    A = torch.matmul(G, old_V)

    L = torch.sum(T_coupling * torch.cdist(X,Y,p=2))
    t = 1 / L
    inner_tol = 1e-3
    Lam0 = torch.zeros(k, k, device=device)
    Dir = get_tagent_direction(old_V, -A, Lam0, t, rho, inner_tol)
    D = Dir['D']
    alpha = 5 # learning rate of V  
    V_new = retraction(old_V, D, alpha, type='svd')
    dist_mat = pdist(X, Y, V_new)
    pwd = torch.sum(T_coupling*dist_mat)
    obj_new = -pwd + rho*torch.linalg.norm(V_new, ord=1) 
    while obj_new > (obj_old*(1-0.001*alpha)):
        alpha = alpha * 0.5
        if alpha < 1e-6:
            break
        V_new = retraction(old_V, D, alpha, type='svd')
        dist_mat = pdist(X, Y, V_new)
        pwd = torch.sum(T_coupling*dist_mat)
        obj_new = -pwd + rho*torch.linalg.norm(V_new, ord=1)   
    return V_new

# @description Obtain the tagent direction (manifold optimization)
# @param Lam: initial value of the Lagrangian multiplier
# @param grad: Eucidean gradient of f at V
# @details regularized semi-smooth Newton method for the following subproblem:
#          min_D <grad f(V), D> + 1/(2t)\|D\|_F^2 + rho \|D+V\|_1
#          s.t. D in Tagent space of V (D^t V + V^t D = 0, Stiefel manifold)
def get_tagent_direction(V, grad, Lam, t, rho, inner_tol, max_iter=50):

    r = V.size(1)
    rdim = int(r*(r+1)/2)
    B_Lam = V - t*(grad-2*torch.matmul(V, Lam))
    Z, De = prox_l1(B_Lam, t*rho) 
    D_Lam = Z - V
    E_Lam =  torch.matmul(V.t(), D_Lam) + torch.matmul(D_Lam.t(), V)
    vec_E = E_Lam.reshape(r**2,1)
    Dn = duplication_matrix(r)
    pDn = torch.linalg.solve(torch.matmul(Dn.t(), Dn), Dn.t())
    vech_E = torch.matmul(pDn, vec_E)
    vec_Lam = Lam.t().reshape(r**2,1) # vectorize by column
    vech_Lam = torch.matmul(pDn, vec_Lam)
    obj_old = torch.linalg.norm(E_Lam, 'fro')
    iter = 0
    while (obj_old**2 > inner_tol) and (iter <= max_iter):
        reg = 0.2*max(min(obj_old, 0.1), 1e-11)
        nnZ = torch.sum(Z!=0)
        # if r < 15:
        if nnZ > rdim:
            C = torch.zeros(r**2, r**2, device=device)
            for i in range(r):
                C[(i*r):((i+1)*r), (i*r):((i+1)*r)] = torch.matmul(torch.matmul(V.t(), torch.diag(De[:,i])), V)

            G = 4*t*torch.matmul(torch.matmul(pDn, C), Dn)
            d = - torch.linalg.solve(G+reg*torch.eye(rdim, device=device), vech_E)
        else:
            Ustack = torch.zeros(nnZ, r**2, device=device)
            dim = 0
            for i in range(r):
                row_idx = torch.nonzero(De[:,i]==1)
                Ustack[dim:(dim+row_idx.shape[0]), (i*r):(i*r+r)] = V[row_idx[:,0], :]
                dim = dim + row_idx.shape[0]
            U = torch.matmul(Ustack, Dn)
            X = 4*t*torch.matmul(pDn, Ustack.t())
            G = torch.eye(nnZ, device=device) + 1/reg * torch.matmul(U, X)
            d = -( 1/reg * vech_E - 1/reg**2 * torch.matmul(X, torch.linalg.solve(G, torch.matmul(U, vech_E))) )
        # else: #r >=15, implement conjugate gradient
        # refer to the public code of ManPG or AManPG 
        # "https://github.com/chenshixiang/ManPG/blob/master/manpg_code_share/SSN_subproblem/Semi_newton_matrix.m"
        vech_Lam_new = vech_Lam + d
        Lam_new = torch.matmul(Dn, vech_Lam_new).reshape(r,r).t()
        B_Lam = V - t*(grad-2*torch.matmul(V, Lam_new))
        Z_new, De_new = prox_l1(B_Lam, t*rho)
        D_Lam = Z_new - V
        E_Lam_new =  torch.matmul(V.t(), D_Lam) + torch.matmul(D_Lam.t(), V)
        vec_E_new = E_Lam_new.reshape(r**2,1)
        vech_E_new = torch.matmul(pDn, vec_E_new)
        obj_new = torch.linalg.norm(E_Lam_new, 'fro')
        lr = 1
        # line search
        while ((obj_new**2) >= (obj_old**2 *(1-0.001*lr))) and (lr > 0.001):
            lr = lr * 0.5
            vech_Lam_new = vech_Lam + lr*d
            Lam_new = torch.matmul(Dn, vech_Lam_new).reshape(r,r).t()
            B_Lam = V - t*(grad-2*torch.matmul(V, Lam_new))
            Z_new, De_new = prox_l1(B_Lam, t*rho)
            D_Lam = Z_new - V
            E_Lam_new =  torch.matmul(V.t(), D_Lam) + torch.matmul(D_Lam.t(), V)
            vec_E_new = E_Lam_new.reshape(r**2,1)
            vech_E_new = torch.matmul(pDn, vec_E_new)
            obj_new = torch.linalg.norm(E_Lam_new, 'fro')
        Z = Z_new.clone()
        De = De_new.clone()
        Lam = Lam_new.clone()
        vech_Lam = vech_Lam_new.clone()
        vech_E = vech_E_new.clone()
        obj_old = obj_new.clone()
        iter = iter + 1
    if iter > max_iter:
        stop_flag = 1
    else:
        stop_flag = 0
    return {'D': D_Lam, 'Lam': Lam, 'stop_flag': stop_flag, 'obj': obj_old}

# @description auxiliary functions

# @description proximal operator
# @details min_X 1/(2t)\|X-B\|^2 + \|X\|_1
def prox_l1(B, t):

    n1, n2 = B.size()
    X = torch.zeros(n1, n2, device=device)
    larger_than_t = B > t
    X = torch.where(larger_than_t, B-t, X)
    smaller_than_minust = B < (-t)
    X = torch.where(smaller_than_minust, B+t, X)
    # X[B>t] = B[B>t] - t 
    # X[B<-t] = B[B<-t] + t
    De = (torch.abs(B) > t).float()
    return X, De

def duplication_matrix(N):
    N_bar = int(N * (N + 1) / 2)
    Dn = torch.zeros((N ** 2, N_bar), device=device)
    for ii in range(N):
        for jj in range(N):
            if jj <= ii:
                u_ij = torch.zeros((N_bar, 1), device=device)
                tmp_idx = int(jj * N + ii + 1 - ((jj + 1) * jj) / 2) - 1
                u_ij[tmp_idx] = 1
                T_ij = torch.zeros((N, N), device=device)
                T_ij[ii, jj] = 1
                T_ij[jj, ii] = 1
                tmp = torch.matmul(u_ij, T_ij.reshape((1, N ** 2)))
                Dn += tmp.t()
    return Dn

# @description retraction operation from tangent space to manifold 
def retraction(V, D, lr, type='svd'):
    r = V.size(1)
    if type=='exp':
        Q, R = torch.linalg.qr(D - torch.matmul(torch.matmul(V, V.t()), D))
        mat1 = torch.cat((V, Q), 1)
        mat21 = torch.cat((torch.matmul(V.t(), D), R), 0)
        mat22 = torch.cat((-R.t(), torch.zeros(r,r, device=device)), 0)
        # mat2 = torch.linalg.matrix_exp(lr*torch.cat((mat21, mat22), 1))
        mat2 = lr*torch.cat((mat21, mat22), 1)
        evals, evectors = torch.linalg.eig(mat2)
        evals = torch.real(evals)
        evectors = torch.real(evectors)
        mat2 = torch.matmul(evectors, torch.matmul(torch.diag(torch.exp(evals)), evectors.t()))
        mat3 = torch.cat((torch.eye(r, device=device), torch.zeros(r,r, device=device)), 0)
        V = torch.matmul(torch.matmul(mat1, mat2), mat3)
    elif type=='polar':
        D = lr*D
        mat1 = V + D
        mat2 = torch.eye(r, device=device) + torch.matmul(D.t(), D)
        evals, evectors = torch.linalg.eig(mat2)
        evals = torch.real(evals)
        evectors = torch.real(evectors)
        evals = 1 / torch.sqrt(evals)
        mat2 = torch.matmul(torch.matmul(evectors, torch.diag(evals)), evectors.t())
        V = torch.matmul(mat1, mat2)
    elif type=='qr':
        Q, R = torch.linalg.qr(V + lr*D)
        V = torch.matmul(Q, torch.diag(torch.diag(torch.sign(R))))
    elif type=='svd':
        mat = V + lr*D
        evals, evectors = torch.linalg.eig(torch.matmul(mat.t(), mat))
        evals = torch.real(evals)
        evectors = torch.real(evectors)
        V = torch.matmul(mat, torch.matmul(torch.matmul(evectors, torch.diag(1/torch.sqrt(evals))), evectors.t()))
    return V


# @description compute the Euclidean distance of projected varaibles
def pdist(X, Y, V):

    x_proj = torch.matmul(X, V)
    y_proj = torch.matmul(Y, V)
    dist_mat = torch.cdist(x_proj, y_proj, p=2)
    return dist_mat

# @description Compute the optimal coupling (transport plan)
# @detail given two marginal distributions (p1, p2) and the projection (V)
def update_OT(X, Y, V, p1, p2):

    dist_mat = pdist(X, Y, V)
    T_coupling = ot.emd(p1, p2, dist_mat)
    return T_coupling

# @description compute the sample covariance matrix
def myCov(data):
    n, p = data.size()
    mean_vec = torch.mean(data, 0).reshape(p,1)
    cov_mat = torch.matmul(data.t(), data) / n - torch.matmul(mean_vec, mean_vec.t())
    return cov_mat

# @description obtain the quantile estimate of the limiting distribution    
def getQuantile_2side(q, m, mc=1000000):
    rvs = torch.randn(mc, m)
    max_rvs, _ = torch.max(torch.abs(rvs), 1)
    test_thresh = torch.quantile(max_rvs, q).numpy()
    return test_thresh