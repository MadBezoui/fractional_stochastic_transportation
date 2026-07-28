import pulp
import numpy as np
from model_builder import build_scenario_subproblem

def run_progressive_hedging(tree, params, rho_PH=1.0, epsilon=1e-3, max_iter=20):
    scenarios = tree.scenarios
    
    # Consensus
    y_bar = {k: 0.0 for k in params.K}
    eta_bar = 0.0
    
    v_bar = {n.node_id: {} for n in tree.nodes}
    # Initialize keys for each node
    for n in tree.nodes:
        nid = n.node_id
        for i in params.I:
            for j in params.J:
                for k in params.K:
                    v_bar[nid][f"x_{i}_{j}_{k}"] = 0.0
        for j in params.J:
            v_bar[nid][f"q_{j}"] = 0.0
            v_bar[nid][f"u_{j}"] = 0.0
            v_bar[nid][f"z_{j}"] = 0.0
            v_bar[nid][f"m_{j}"] = 0.0
        v_bar[nid]["b"] = 0.0
        v_bar[nid]["s"] = 0.0
        
    # Multipliers
    mu_y = {s['omega']: {k: 0.0 for k in params.K} for s in scenarios}
    mu_eta = {s['omega']: 0.0 for s in scenarios}
    
    mu_v = {s['omega']: {n.node_id: {k: 0.0 for k in v_bar[n.node_id]} for n in s['path']} for s in scenarios}
    
    history = []
    
    for r in range(max_iter):
        y_omega = {s['omega']: {} for s in scenarios}
        eta_omega = {}
        v_omega = {s['omega']: {n.node_id: {} for n in s['path']} for s in scenarios}
        
        # a. Solve scenario subproblems
        for omega_dict in scenarios:
            omega = omega_dict['omega']
            model, y, x, r_var, q, u, z, m, b, s_var, eta, xi = build_scenario_subproblem(
                omega_dict, params, y_bar, eta_bar, mu_y[omega], mu_eta[omega], v_bar, mu_v[omega], rho_PH
            )
            
            model.solve(pulp.GUROBI(msg=False))
            
            for k in params.K:
                y_omega[omega][k] = pulp.value(y[k])
            eta_omega[omega] = pulp.value(eta)
            
            for node in omega_dict['path']:
                nid = node.node_id
                for i in params.I:
                    for j in params.J:
                        for k in params.K:
                            v_omega[omega][nid][f"x_{i}_{j}_{k}"] = pulp.value(x[i, j, k, nid])
                for j in params.J:
                    v_omega[omega][nid][f"q_{j}"] = pulp.value(q[j, nid])
                    v_omega[omega][nid][f"u_{j}"] = pulp.value(u[j, nid])
                    v_omega[omega][nid][f"z_{j}"] = pulp.value(z[j, nid])
                    v_omega[omega][nid][f"m_{j}"] = pulp.value(m[j, nid])
                v_omega[omega][nid]["b"] = pulp.value(b[nid])
                v_omega[omega][nid]["s"] = pulp.value(s_var[nid])
            
        # b. Compute consensus
        new_y_bar = {k: 0.0 for k in params.K}
        new_eta_bar = 0.0
        new_v_bar = {n.node_id: {k: 0.0 for k in v_bar[n.node_id]} for n in tree.nodes}
        node_probs = {n.node_id: 0.0 for n in tree.nodes}
        
        for omega_dict in scenarios:
            prob = omega_dict['prob']
            for n in omega_dict['path']:
                node_probs[n.node_id] += prob
                
        for omega_dict in scenarios:
            omega = omega_dict['omega']
            prob = omega_dict['prob']
            for k in params.K:
                new_y_bar[k] += prob * y_omega[omega][k]
            new_eta_bar += prob * eta_omega[omega]
            
            for n in omega_dict['path']:
                nid = n.node_id
                weight = prob / node_probs[nid]
                for key in new_v_bar[nid]:
                    new_v_bar[nid][key] += weight * v_omega[omega][nid][key]
            
        # c. Update multipliers and compute residual
        r_nac = 0.0
        for omega_dict in scenarios:
            omega = omega_dict['omega']
            prob = omega_dict['prob']
            
            scenario_diff_sq = 0.0
            
            for k in params.K:
                diff = y_omega[omega][k] - new_y_bar[k]
                scenario_diff_sq += diff**2
                mu_y[omega][k] += rho_PH * diff
                
            diff_eta = eta_omega[omega] - new_eta_bar
            scenario_diff_sq += diff_eta**2
            mu_eta[omega] += rho_PH * diff_eta
            
            for n in omega_dict['path']:
                nid = n.node_id
                for key in new_v_bar[nid]:
                    diff_v = v_omega[omega][nid][key] - new_v_bar[nid][key]
                    scenario_diff_sq += diff_v**2
                    mu_v[omega][nid][key] += rho_PH * diff_v
            
            r_nac += prob * scenario_diff_sq
            
        r_nac = np.sqrt(r_nac)
        
        r_proxy = 0.0
        for k in params.K:
            r_proxy += (rho_PH * (new_y_bar[k] - y_bar[k]))**2
        r_proxy += (rho_PH * (new_eta_bar - eta_bar))**2
        
        for nid in new_v_bar:
            for key in new_v_bar[nid]:
                r_proxy += (rho_PH * (new_v_bar[nid][key] - v_bar[nid][key]))**2 * node_probs[nid]
                
        r_proxy = np.sqrt(r_proxy)

        y_bar = new_y_bar
        eta_bar = new_eta_bar
        v_bar = new_v_bar
        
        history.append({
            'r_proxy': r_proxy,
            'iteration': r,
            'r_nac': r_nac,
            'y_bar': dict(y_bar),
            'eta_bar': eta_bar
        })
        
        if r_nac <= epsilon and r_proxy <= epsilon:
            history[-1]['stopping_reason'] = 'converged'
            break
            
    if len(history) == max_iter and history[-1].get('stopping_reason') != 'converged':
        history[-1]['stopping_reason'] = 'maximum iterations'

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
