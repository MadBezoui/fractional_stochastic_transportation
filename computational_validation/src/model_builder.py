import pulp
import numpy as np
import scipy.special

def get_l1_coefficients(alpha, N, delta_tau, rho_j):
    if alpha == 1.0:
        theta = 1.0 / (1.0 + rho_j * delta_tau)
        return None, None, theta
        
    g_alpha = scipy.special.gamma(2 - alpha) * (delta_tau ** alpha)
    
    a = np.zeros(N + 1)
    for ell in range(N + 1):
        a[ell] = (ell + 1)**(1 - alpha) - ell**(1 - alpha)
        
    b = np.zeros(N + 1)
    for s in range(1, N + 1):
        b[s] = a[s-1] - a[s]
        
    return g_alpha, b, None

class TransportationModelParams:
    def __init__(self):
        self.I = [1]
        self.J = [1, 2]
        self.K = [1, 2]
        
        self.f = {1: 1000.0, 2: 200.0}
        self.v = {}
        self.e = {}
        for i in self.I:
            for j in self.J:
                self.v[(i, j, 1)] = 5.0
                self.e[(i, j, 1)] = 0.2
                self.v[(i, j, 2)] = 15.0
                self.e[(i, j, 2)] = 0.8
        self.h = {j: 2 for j in self.J}
        self.pi = {j: 500 for j in self.J}
        
        self.S = {i: 100 for i in self.I}
        self.Q = {k: 50 for k in self.K}
        self.delta = {j: 0.05 for j in self.J}
        
        self.alpha = 0.8
        self.rho = {j: 0.1 for j in self.J}
        self.psi = {j: 0.5 for j in self.J}
        self.chi = {j: 0.5 for j in self.J}
        self.delta_tau = 1.0
        
        self.m_bar = {j: 1000 for j in self.J}
        self.gamma = {j: 0.1 for j in self.J}
        self.z_0 = {j: 0 for j in self.J}
        
        self.E_cap = 10
        self.B_bar = 50
        self.S_bar = 50
        self.P_B = 10
        self.P_S = 8
        
        self.lambd = 0.5
        self.beta = 0.95

def build_deterministic_equivalent(tree, params: TransportationModelParams):
    model = pulp.LpProblem("Fractional_Transportation_DE", pulp.LpMinimize)
    
    y = pulp.LpVariable.dicts("y", params.K, cat=pulp.LpBinary)
    
    x = pulp.LpVariable.dicts("x", ((i, j, k, n.node_id) for i in params.I for j in params.J for k in params.K for n in tree.nodes), lowBound=0)
    r = pulp.LpVariable.dicts("r", ((j, n.node_id) for j in params.J for n in tree.nodes), lowBound=0)
    q = pulp.LpVariable.dicts("q", ((j, n.node_id) for j in params.J for n in tree.nodes), lowBound=0)
    u = pulp.LpVariable.dicts("u", ((j, n.node_id) for j in params.J for n in tree.nodes), lowBound=0)
    z = pulp.LpVariable.dicts("z", ((j, n.node_id) for j in params.J for n in tree.nodes), lowBound=0)
    m = pulp.LpVariable.dicts("m", ((j, n.node_id) for j in params.J for n in tree.nodes), lowBound=-params.m_bar[1]*10, upBound=params.m_bar[1]*10)
    
    b = pulp.LpVariable.dicts("b", (n.node_id for n in tree.nodes), lowBound=0, upBound=params.B_bar)
    s = pulp.LpVariable.dicts("s", (n.node_id for n in tree.nodes), lowBound=0, upBound=params.S_bar)
    
    eta = pulp.LpVariable("eta", cat=pulp.LpContinuous)
    xi = pulp.LpVariable.dicts("xi", (omega['omega'] for omega in tree.scenarios), lowBound=0)
    
    N = max(n.t for n in tree.nodes)
    
    for n in tree.nodes:
        if n.t == 0:
            model += b[n.node_id] == 0
            model += s[n.node_id] == 0
            for i in params.I:
                for j in params.J:
                    for k in params.K:
                        model += x[i, j, k, n.node_id] == 0
        
        for i in params.I:
            model += pulp.lpSum(x[i, j, k, n.node_id] for j in params.J for k in params.K) <= params.S[i]
            
        for k in params.K:
            model += pulp.lpSum(x[i, j, k, n.node_id] for i in params.I for j in params.J) <= params.Q[k] * y[k]
            
        for j in params.J:
            model += r[j, n.node_id] == pulp.lpSum(x[i, j, k, n.node_id] for i in params.I for k in params.K)
            model += q[j, n.node_id] + u[j, n.node_id] == n.demand.get(j, 0)
            
            if n.parent is None:
                model += z[j, n.node_id] == params.z_0[j] + r[j, n.node_id] - q[j, n.node_id]
                model += q[j, n.node_id] <= r[j, n.node_id]
                model += m[j, n.node_id] == 0
            else:
                p_id = n.parent.node_id
                model += z[j, n.node_id] == (1 - params.delta[j]) * z[j, p_id] + r[j, n.node_id] - q[j, n.node_id]
                
                model += q[j, n.node_id] <= (1 - params.delta[j]) * z[j, p_id] + r[j, n.node_id]
                model += q[j, n.node_id] + params.chi[j] * m[j, p_id] <= (1 - params.delta[j]) * z[j, p_id] + r[j, n.node_id]
                
                F_t = params.psi[j] * (r[j, n.node_id] - q[j, n.node_id])
                
                if params.alpha == 1.0:
                    _, _, theta = get_l1_coefficients(params.alpha, N, params.delta_tau, params.rho[j])
                    model += m[j, n.node_id] == theta * (m[j, p_id] + params.delta_tau * F_t)
                else:
                    g_alpha, b_coeff, _ = get_l1_coefficients(params.alpha, N, params.delta_tau, params.rho[j])
                    path = n.get_path_to_root()
                    t = n.t
                    past_sum = 0
                    for ell in range(1, t):
                        node_ell = path[ell]
                        past_sum += b_coeff[t - ell] * m[j, node_ell.node_id]
                    model += (1 + g_alpha * params.rho[j]) * m[j, n.node_id] == g_alpha * F_t + past_sum
                    
            model += m[j, n.node_id] <= params.m_bar[j]
            model += m[j, n.node_id] >= -params.m_bar[j]
            
        if n.t > 0:
            E_t = pulp.lpSum(params.e[i, j, k] * x[i, j, k, n.node_id] for i in params.I for j in params.J for k in params.K)
            model += E_t <= params.E_cap + b[n.node_id] - s[n.node_id]
        
    for scenario in tree.scenarios:
        for j in params.J:
            for t_idx in range(len(scenario['path'])):
                cum_q = pulp.lpSum(q[j, scenario['path'][tau].node_id] for tau in range(t_idx + 1))
                cum_d = sum(scenario['path'][tau].demand.get(j, 0) for tau in range(t_idx + 1))
                model += cum_q >= params.gamma[j] * cum_d

    fixed_cost = pulp.lpSum(params.f[k] * y[k] for k in params.K)
    
    scenario_costs = {}
    for scenario in tree.scenarios:
        omega = scenario['omega']
        cost_omega = 0
        for node in scenario['path']:
            n_id = node.node_id
            op_cost = pulp.lpSum(params.v[i, j, k] * x[i, j, k, n_id] for i in params.I for j in params.J for k in params.K)
            inv_cost = pulp.lpSum(params.h[j] * z[j, n_id] for j in params.J)
            pen_cost = pulp.lpSum(params.pi[j] * u[j, n_id] for j in params.J)
            carb_cost = params.P_B * b[n_id] - params.P_S * s[n_id]
            cost_omega += op_cost + inv_cost + pen_cost + carb_cost
            
        scenario_costs[omega] = cost_omega
        model += xi[omega] >= cost_omega - eta
        
    expected_cost = pulp.lpSum(scenario['prob'] * scenario_costs[scenario['omega']] for scenario in tree.scenarios)
    cvar_term = eta + (1.0 / (1.0 - params.beta)) * pulp.lpSum(scenario['prob'] * xi[scenario['omega']] for scenario in tree.scenarios)
    
    obj = fixed_cost + (1 - params.lambd) * expected_cost + params.lambd * cvar_term
    model += obj
    
    return model, y, x, r, q, u, z, m, b, s, eta, xi

def build_scenario_subproblem(omega_dict, params: TransportationModelParams, 
                              y_bar, eta_bar, 
                              mu_y, mu_eta, 
                              v_bar, mu_v,
                              rho_PH):
    model = pulp.LpProblem(f"Fractional_PH_Subproblem_{omega_dict['omega']}", pulp.LpMinimize)
    
    y = pulp.LpVariable.dicts("y", params.K, lowBound=0, upBound=1)
    for k in params.K:
        y[k].cat = pulp.LpBinary
        
    path = omega_dict['path']
    nodes = [n.node_id for n in path]
    
    x = pulp.LpVariable.dicts("x", ((i, j, k, n) for i in params.I for j in params.J for k in params.K for n in nodes), lowBound=0)
    r = pulp.LpVariable.dicts("r", ((j, n) for j in params.J for n in nodes), lowBound=0)
    q = pulp.LpVariable.dicts("q", ((j, n) for j in params.J for n in nodes), lowBound=0)
    u = pulp.LpVariable.dicts("u", ((j, n) for j in params.J for n in nodes), lowBound=0)
    z = pulp.LpVariable.dicts("z", ((j, n) for j in params.J for n in nodes), lowBound=0)
    m = pulp.LpVariable.dicts("m", ((j, n) for j in params.J for n in nodes), lowBound=-params.m_bar[1]*10, upBound=params.m_bar[1]*10)
    
    b = pulp.LpVariable.dicts("b", nodes, lowBound=0, upBound=params.B_bar)
    s = pulp.LpVariable.dicts("s", nodes, lowBound=0, upBound=params.S_bar)
    
    eta = pulp.LpVariable("eta", cat=pulp.LpContinuous)
    xi = pulp.LpVariable("xi", lowBound=0)
    
    N = path[-1].t
    
    for idx, node in enumerate(path):
        n = node.node_id
        
        if node.t == 0:
            model += b[n] == 0
            model += s[n] == 0
            for i in params.I:
                for j in params.J:
                    for k in params.K:
                        model += x[i, j, k, n] == 0
                        
        for i in params.I:
            model += pulp.lpSum(x[i, j, k, n] for j in params.J for k in params.K) <= params.S[i]
            
        for k in params.K:
            model += pulp.lpSum(x[i, j, k, n] for i in params.I for j in params.J) <= params.Q[k] * y[k]
            
        for j in params.J:
            model += r[j, n] == pulp.lpSum(x[i, j, k, n] for i in params.I for k in params.K)
            model += q[j, n] + u[j, n] == node.demand.get(j, 0)
            
            if idx == 0:
                model += z[j, n] == params.z_0[j] + r[j, n] - q[j, n]
                model += q[j, n] <= r[j, n]
                model += m[j, n] == 0
            else:
                p = path[idx-1].node_id
                model += z[j, n] == (1 - params.delta[j]) * z[j, p] + r[j, n] - q[j, n]
                model += q[j, n] <= (1 - params.delta[j]) * z[j, p] + r[j, n]
                model += q[j, n] + params.chi[j] * m[j, p] <= (1 - params.delta[j]) * z[j, p] + r[j, n]
                
                F_t = params.psi[j] * (r[j, n] - q[j, n])
                if params.alpha == 1.0:
                    _, _, theta = get_l1_coefficients(params.alpha, N, params.delta_tau, params.rho[j])
                    model += m[j, n] == theta * (m[j, p] + params.delta_tau * F_t)
                else:
                    g_alpha, b_coeff, _ = get_l1_coefficients(params.alpha, N, params.delta_tau, params.rho[j])
                    past_sum = 0
                    for ell in range(1, node.t):
                        node_ell = path[ell].node_id
                        past_sum += b_coeff[node.t - ell] * m[j, node_ell]
                    model += (1 + g_alpha * params.rho[j]) * m[j, n] == g_alpha * F_t + past_sum
                    
            model += m[j, n] <= params.m_bar[j]
            model += m[j, n] >= -params.m_bar[j]
            
        if node.t > 0:
            E_t = pulp.lpSum(params.e[i, j, k] * x[i, j, k, n] for i in params.I for j in params.J for k in params.K)
            model += E_t <= params.E_cap + b[n] - s[n]
        
    for j in params.J:
        for t_idx in range(len(path)):
            cum_q = pulp.lpSum(q[j, path[tau].node_id] for tau in range(t_idx + 1))
            cum_d = sum(path[tau].demand.get(j, 0) for tau in range(t_idx + 1))
            model += cum_q >= params.gamma[j] * cum_d

    cost_omega = 0
    for node in path:
        n = node.node_id
        cost_omega += pulp.lpSum(params.v[i, j, k] * x[i, j, k, n] for i in params.I for j in params.J for k in params.K)
        cost_omega += pulp.lpSum(params.h[j] * z[j, n] for j in params.J)
        cost_omega += pulp.lpSum(params.pi[j] * u[j, n] for j in params.J)
        cost_omega += params.P_B * b[n] - params.P_S * s[n]
        
    model += xi >= cost_omega - eta
    
    phi_tilde = pulp.lpSum(params.f[k] * y[k] for k in params.K) + \
                (1 - params.lambd) * cost_omega + \
                params.lambd * eta + \
                (params.lambd / (1 - params.beta)) * xi
                
    # Add PWL penalties
    penalty = 0
    
    # Binary variables
    for k in params.K:
        y_val = float(y_bar[k])
        penalty += mu_y[k] * y[k] + (rho_PH / 2.0) * ((1 - 2 * y_val) * y[k] + y_val**2)
        
    # Helper for continuous PWL penalty
    U_bound = 10.0
    L_segs = 10
    kappa = [ell * (U_bound / L_segs) for ell in range(L_segs + 1)]
    
    def add_pwl_penalty(var, var_bar, mu, s_scale, name_prefix):
        nonlocal penalty
        nonlocal model
        u_var = pulp.LpVariable(f"{name_prefix}_u", lowBound=0)
        u_var_scaled = pulp.LpVariable(f"{name_prefix}_u_scaled", lowBound=0, upBound=U_bound)
        
        model += u_var >= var - var_bar
        model += u_var >= var_bar - var
        model += u_var_scaled == (1.0 / s_scale) * u_var
        
        g_var = pulp.LpVariable(f"{name_prefix}_g", lowBound=0)
        for ell in range(1, L_segs + 1):
            model += g_var >= (kappa[ell-1] + kappa[ell]) * u_var_scaled - kappa[ell-1] * kappa[ell]
            
        penalty += mu * var + (rho_PH / 2.0) * (g_var * (s_scale ** 2))
        
    # Penalty for eta
    add_pwl_penalty(eta, eta_bar, mu_eta, 100.0, "eta")
    
    # Penalties for operational variables
    # s_h scales
    scales = {
        'x': 100.0, 'q': 100.0, 'u': 100.0, 'z': 100.0, 
        'm': 1000.0, 'b': 50.0, 's': 50.0
    }
    
    for n in nodes:
        if n not in v_bar:
            continue
            
        # x
        for i in params.I:
            for j in params.J:
                for k in params.K:
                    k_str = f"x_{i}_{j}_{k}"
                    if k_str in v_bar[n]:
                        add_pwl_penalty(x[i, j, k, n], v_bar[n][k_str], mu_v[n][k_str], scales['x'], f"x_{i}_{j}_{k}_{n}")
                        
        # q, u, z, m
        for j in params.J:
            for key, var_dict in [('q', q), ('u', u), ('z', z), ('m', m)]:
                k_str = f"{key}_{j}"
                if k_str in v_bar[n]:
                    add_pwl_penalty(var_dict[j, n], v_bar[n][k_str], mu_v[n][k_str], scales[key], f"{k_str}_{n}")
                    
        # b, s
        if 'b' in v_bar[n]:
            add_pwl_penalty(b[n], v_bar[n]['b'], mu_v[n]['b'], scales['b'], f"b_{n}")
        if 's' in v_bar[n]:
            add_pwl_penalty(s[n], v_bar[n]['s'], mu_v[n]['s'], scales['s'], f"s_{n}")
            
    model += phi_tilde + penalty
    
    return model, y, x, r, q, u, z, m, b, s, eta, xi
