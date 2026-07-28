import pulp
import numpy as np
from model_builder import build_scenario_subproblem

def run_progressive_hedging(tree, params, rho_PH=1.0, epsilon=1e-3, max_iter=20):
    # 1. Initialization
    scenarios = tree.scenarios
    
    # Consensus
    y_bar = {k: 0.0 for k in params.K}
    eta_bar = 0.0
    
    # Multipliers
    mu_y = {s['omega']: {k: 0.0 for k in params.K} for s in scenarios}
    mu_eta = {s['omega']: 0.0 for s in scenarios}
    
    # Logging
    history = []
    
    # 2. Main loop
    for r in range(max_iter):
        y_omega = {s['omega']: {} for s in scenarios}
        eta_omega = {}
        
        # a. Solve scenario subproblems
        for omega_dict in scenarios:
            omega = omega_dict['omega']
            model, y, x, req, q, u, z, m, b, s_var, eta, xi = build_scenario_subproblem(
                omega_dict, params, y_bar, eta_bar, mu_y[omega], mu_eta[omega], rho_PH
            )
            
            # Using CBC solver (default for PuLP)
            model.solve(pulp.GUROBI(msg=False))
            
            if model.status != pulp.LpStatusOptimal:
                print(f"Warning: Scenario {omega} subproblem not optimal. Status: {pulp.LpStatus[model.status]}")
            
            for k in params.K:
                y_omega[omega][k] = pulp.value(y[k])
            eta_omega[omega] = pulp.value(eta)
            
        # b. Compute consensus
        new_y_bar = {k: 0.0 for k in params.K}
        new_eta_bar = 0.0
        
        for omega_dict in scenarios:
            omega = omega_dict['omega']
            prob = omega_dict['prob']
            for k in params.K:
                new_y_bar[k] += prob * y_omega[omega][k]
            new_eta_bar += prob * eta_omega[omega]
            
        # c. Update multipliers and compute residual
        r_nac = 0.0
        for omega_dict in scenarios:
            omega = omega_dict['omega']
            prob = omega_dict['prob']
            
            # NAC residual calculation (L2 norm)
            scenario_diff_sq = 0.0
            
            for k in params.K:
                diff = y_omega[omega][k] - new_y_bar[k]
                scenario_diff_sq += diff**2
                mu_y[omega][k] += rho_PH * diff
                
            diff_eta = eta_omega[omega] - new_eta_bar
            scenario_diff_sq += diff_eta**2
            mu_eta[omega] += rho_PH * diff_eta
            
            r_nac += prob * scenario_diff_sq
            
        r_nac = np.sqrt(r_nac)
        
        # Update consensus for next iteration
        # Compute R_proxy (dual residual)
        r_proxy = 0.0
        for k in params.K:
            r_proxy += (rho_PH * (new_y_bar[k] - y_bar[k]))**2
        r_proxy += (rho_PH * (new_eta_bar - eta_bar))**2
        r_proxy = np.sqrt(r_proxy)

        y_bar = new_y_bar
        eta_bar = new_eta_bar
        
        history.append({
            'r_proxy': r_proxy,
            'iteration': r,
            'r_nac': r_nac,
            'y_bar': dict(y_bar),
            'eta_bar': eta_bar
        })
        
        if r_nac <= epsilon:
            break
            

    # 3. Recover policy
    from model_builder import build_deterministic_equivalent
    import itertools
    
    delta_fix = 0.1
    y_hat_base = {k: (1 if y_bar[k] >= 0.5 else 0) for k in params.K}
    
    ambiguous = [k for k in params.K if abs(y_bar[k] - 0.5) <= delta_fix]
    candidates = []
    
    if not ambiguous:
        candidates.append(y_hat_base)
    else:
        # Generate all permutations for ambiguous variables
        permutations = list(itertools.product([0, 1], repeat=len(ambiguous)))
        for perm in permutations:
            cand = dict(y_hat_base)
            for i, k in enumerate(ambiguous):
                cand[k] = perm[i]
            candidates.append(cand)
            
    best_cand = None
    best_obj = float('inf')
    
    for cand in candidates:
        model_de, y_de, *_ = build_deterministic_equivalent(tree, params)
        for k in params.K:
            y_de[k].setInitialValue(cand[k])
            y_de[k].fixValue()
        
        model_de.solve(pulp.GUROBI(msg=False))
        if model_de.status == pulp.LpStatusOptimal:
            obj_val = pulp.value(model_de.objective)
            if obj_val < best_obj:
                best_obj = obj_val
                best_cand = cand
                
    if best_cand is None:
        y_hat = y_hat_base
    else:
        y_hat = best_cand
        
    return y_hat, eta_bar, history

