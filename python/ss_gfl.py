import numpy as np


from numpy.ctypeslib import ndpointer
from ctypes import *

graphfl_lib = cdll.LoadLibrary('libgraphfl.so')
weighted_fl = graphfl_lib.tf_dp_weight
weighted_fl.restype = c_int

weighted_fl.argtypes = [c_int, # int n
                        ndpointer(c_double, flags='C_CONTIGUOUS'), # double *y
                        ndpointer(c_double, flags='C_CONTIGUOUS'), # double *w
                        c_double, # double lam
                        ndpointer(c_double, flags='C_CONTIGUOUS'), # double *beta
                        ndpointer(c_double, flags='C_CONTIGUOUS'), # double *x
                        ndpointer(c_double, flags='C_CONTIGUOUS'), # double *a
                        ndpointer(c_double, flags='C_CONTIGUOUS'), # double *b
                        ndpointer(c_double, flags='C_CONTIGUOUS'), # double *tm
                        ndpointer(c_double, flags='C_CONTIGUOUS'), # double *tp
                        ]

def solve_weighted_gfl(y, lams, beta_init=None, rel_tol=1e-6, alpha=2, max_admm_steps=10000, **kwargs):
    if beta_init is None:
        beta = np.full(y.shape[0], y.mean())
    else:
        beta = beta_init.copy()

    # Get the slack variables
    z = np.array(np.repeat(beta, 2)[1:-1])
    u = np.zeros_like(z)
    weights = np.array([alpha/2, alpha/2], dtype='double')
    
    # Simple buffers for the C function
    z_buf = np.zeros(2, dtype='double')
    x_buf = np.zeros(4, dtype='double')
    a_buf = np.zeros(4, dtype='double')
    b_buf = np.zeros(4, dtype='double')
    tm_buf = np.zeros(2, dtype='double')
    tp_buf = np.zeros(2, dtype='double')
    
    # Run ADMM until convergence
    prev = beta.copy()
    for step in range(max_admm_steps):
        # Beta step
        beta[0] = (2 * y[0] + alpha*(z[0] - u[0])) / (2 + alpha)
        beta[-1] = (2 * y[-1] + alpha*(z[-1] - u[-1])) / (2 + alpha)
        beta[1:-1] = (2 * y[1:-1] + alpha * (z[1:-1] - u[1:-1]).reshape((-1,2)).sum(axis=1)) / (2 + 2*alpha)

        # Run the 1D FL for every edge
        for i in range(y.shape[0]-1):
            weighted_fl(2, beta[i:i+2] + u[2*i:2*i+2], weights, lams[i], z_buf, x_buf, a_buf, b_buf, tm_buf, tp_buf)
            z[2*i:2*i+2] = z_buf

        # Update the dual variable
        u += np.repeat(beta, 2)[1:-1] - z

        # Check for convergence (not recommended this way, but it's fast)
        delta = np.linalg.norm(prev - beta)
        if delta <= rel_tol:
            break
        
        prev = beta.copy()

    return beta

class FastWeightedFusedLassoSolver:
    def __init__(self, y):
        # Pre-cache a sparse LU decomposition of the FL matrix
        from pygfl.utils import get_1d_penalty_matrix
        from scipy.sparse.linalg import factorized
        from scipy.sparse import csc_matrix
        D = get_1d_penalty_matrix(y.shape[0])
        D = np.vstack([D, np.zeros(y.shape[0])])
        D[-1,-1] = 1e-6 # Nugget for full rank matrix
        D = csc_matrix(D)
        self.invD = factorized(D)

        # Setup the fast GFL solver
        from pygfl.solver import TrailSolver
        from pygfl.trails import decompose_graph
        from pygfl.utils import hypercube_edges, chains_to_trails
        from networkx import Graph
        edges = hypercube_edges(y.shape)
        g = Graph()
        g.add_edges_from(edges)
        chains = decompose_graph(g, heuristic='greedy')
        ntrails, trails, breakpoints, edges = chains_to_trails(chains)
        self.solver = TrailSolver()
        self.solver.set_data(y, edges, ntrails, trails, breakpoints)

        from pygfl.easy import solve_gfl
        self.beta = solve_gfl(y)

    def solve(self, lams, rel_tol=1e-6, alpha=2, max_admm_steps=100000):
        ''' TODO 
        # Run ADMM until convergence
        prev = beta.copy()
        for step in range(max_admm_steps):
            # Beta step
            beta[0] = (2 * y[0] + alpha*(z[0] - u[0])) / (2 + alpha)
            beta[-1] = (2 * y[-1] + alpha*(z[-1] - u[-1])) / (2 + alpha)
            beta[1:-1] = (2 * y[1:-1] + alpha * (z[1:-1] - u[1:-1]).reshape((-1,2)).sum(axis=1)) / (2 + 2*alpha)

            # Run the 1D FL for every edge
            for i in range(y.shape[0]-1):
                weighted_fl(2, beta[i:i+2] + u[2*i:2*i+2], weights, lams[i], z_buf, x_buf, a_buf, b_buf, tm_buf, tp_buf)
                z[2*i:2*i+2] = z_buf

            # Update the dual variable
            u += np.repeat(beta, 2)[1:-1] - z

            # Check for convergence (not recommended this way, but it's fast)
            delta = np.linalg.norm(prev - beta)
            if delta <= rel_tol:
                break
            
            prev = beta.copy()

        return beta
        '''
        pass

def estimate_hyperparams(y):
    '''Efron-style two-groups estimate of the sparsity level'''
    from smoothfdr.normix import empirical_null, predictive_recursion
    diffs = y[1:] - y[:-1]
    mu, sigma = empirical_null(diffs)
    noise_variance = sigma**2 / 2
    diffs_range = diffs.max() - diffs.min()
    pr = predictive_recursion(diffs, 10, np.linspace(diffs.min()-0.1*diffs_range, diffs.max()+0.1*diffs_range, 200), mu0=mu, sig0=sigma)
    a, b = pr['pi0'] * diffs.shape[0], (1-pr['pi0']) * diffs.shape[0]
    return noise_variance, a, b

def ss_gfl(y, min_spike=1e-4, max_spike=1e2, nspikes=30, max_steps=100, rel_tol=1e-6, a=None, b=None, **kwargs):
    # if a is None:
    #     a = np.sqrt(len(y))
    # if b is None:
    #     b = np.sqrt(len(y))
    sigma, a, b = estimate_hyperparams(y)
    print('sigma: {} a: {} b: {} expected proportion of nulls: {:.2f}'.format(sigma, a, b, a / (a+b)))
    
    # Create the log-space grid of spikes
    spike_grid = np.exp(np.linspace(np.log(min_spike), np.log(max_spike), nspikes))

    # Initialize beta at the observations
    # beta = y.copy()
    from pygfl.easy import solve_gfl
    beta = solve_gfl(y)

    # Use an equal weighted mixture to start, with two identical slabs
    # diffs = np.abs(beta[1:] - beta[:-1])
    # theta = (a + (diffs < 1e-3).sum()) / (a + b + beta.shape[0] - 3)
    theta = 0.5
    slab = min_spike

    # Track convergence
    prev = beta.copy()

    # Track the BIC path
    bic = np.zeros(nspikes)

    # Create a fast solver for the 1d fused lasso
    fl_solver = FastWeightedFusedLassoSolver(y)

    # Run over the entire solution path of spikes, starting from very flat
    # spikes and going to very sharp spikes
    betas = np.zeros((nspikes, y.shape[0]))
    for spike_idx, spike in enumerate(spike_grid):
        print('Spike {}/{}: {:.4f}'.format(spike_idx+1, nspikes, spike))
        # Run the EM algorithm for the fixed spike, using the warm-started
        # beta and theta values
        for step in range(max_steps):
            print('\tStep {}'.format(step))
            # Get the beta differences
            diffs = np.abs(beta[1:] - beta[:-1])
            # np.set_printoptions(suppress=True, precision=2)
            # print(diffs)

            # E-step: Expected local mixture probabilities given theta and beta
            spike_prob = theta * np.exp(-diffs * spike) * spike
            slab_prob = (1-theta) * np.exp(-diffs * slab) * slab
            gamma = spike_prob / (spike_prob + slab_prob)

            # We have a 2-part M-step.
            # (i) M-step for beta: run the fused lasso with edge weights:
            #           lam_ij = gamma_ij*spike + (1-gamma_ij)*slab
            lams = gamma*spike + (1-gamma)*slab
            beta = solve_weighted_gfl(y, lams * sigma, beta_init=beta, **kwargs)

            # (ii) M-step for theta: MLE prior mixture probabilities
            theta = (a + gamma.sum()) / (a + b + beta.shape[0] - 3)
            theta = theta.clip(1e-3,1-1e-3) # Don't let theta get too big. Equivalent to choosing a and b proportional to the number of nodes
            print('\ttheta: {:.3f}'.format(theta))

            # Check for convergence
            delta = np.linalg.norm(prev - beta)
            if delta <= rel_tol:
                print()
                break

            print('\tDelta={:.6f}'.format(delta))
            print()
            prev = beta.copy()

        # Calculate BIC = -2ln(L) + dof * (ln(n) - ln(2pi))
        nll = -0.5 / sigma * ((y - beta)**2).sum()
        dof = (np.abs(beta[1:] - beta[:-1]) >= 1e-4).sum() + 1
        bic[spike_idx] = 2*nll + dof * (np.log(beta.shape[0]) - np.log(2 * np.pi))
        print('NLL: {:.4f} dof: {} BIC: {:.2f}'.format(nll, dof, bic[spike_idx]))

        # if spike_idx > 0 and np.abs(bic[spike_idx] - bic[spike_idx-1]) <= rel_tol:
        #     break

        # Save the entire path of solutions
        betas[spike_idx] = beta
        

    return {'betas': betas[:spike_idx], 'bic': bic[:spike_idx]}

if __name__ == '__main__':
    from pygfl.easy import solve_gfl
    import matplotlib.pyplot as plt
    np.random.seed(5)
    truth = np.array([0]*20 + [4]*30 + [-1]*40 + [-5]*20 + [-1]*30 + [1.8]*10 + [-0.8]*30 + [3]*50)
    X = np.arange(1,1+len(truth))
    Y = np.random.normal(truth)

    # Fit the ordinary GFL
    beta_gfl = solve_gfl(Y)

    # Fit the spike-and-slab GFL
    results = ss_gfl(Y)

    # Use the last solution
    beta_ssl = results['betas'][-1]

    # Use the BIC solution
    # beta_ssl_bic = results['betas'][np.argmin(results['bic'])]

    plt.scatter(X, Y, color='gray', alpha=0.2, label='Observations')
    plt.plot(X, truth, color='black', label='Truth')
    plt.plot(X, beta_gfl, color='blue', label='FL')
    plt.plot(X, beta_ssl, color='orange', label='SSFL')
    # plt.plot(X, beta_ssl_bic, color='green', label='SSFL (BIC)')
    plt.legend(loc='lower right')

    plt.savefig('plots/gfl.pdf', bbox_inches='tight')
    plt.close()

    print('MSE')
    print('GFL: {}'.format(np.mean((truth - beta_gfl)**2)))
    print('SSFL (last): {}'.format(np.mean((truth - beta_ssl)**2)))
    # print('SSFL (BIC): {}'.format(np.mean((truth - beta_ssl_bic)**2)))
    print()
    print('Median Absolute Error')
    print('GFL: {}'.format(np.median(np.abs(truth - beta_gfl))))
    print('SSFL (last): {}'.format(np.median(np.abs(truth - beta_ssl))))
    print()
    print('Max error')
    print('GFL: {}'.format(np.abs(truth - beta_gfl).max()))
    print('SSFL (last): {}'.format(np.abs(truth - beta_ssl).max()))
    # print('SSFL (BIC): {}'.format(np.abs(truth - beta_ssl_bic).max()))




