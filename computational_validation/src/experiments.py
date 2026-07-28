import numpy as np
import pulp
import scipy.optimize
import matplotlib.pyplot as plt

from scenario_tree import generate_toy_tree
from model_builder import TransportationModelParams, build_deterministic_equivalent, get_l1_coefficients
from progressive_hedging import run_progressive_hedging

def experiment_1_calibration():
    print("--- Experiment 1: Parameter Identification ---")
    
    # Target params
    alpha_true = 0.8
    rho_true = 0.1
    psi_true = 0.5
    N = 20
    delta_tau = 1.0
    
    # Generate synthetic input F_t (forcing)
    np.random.seed(42)
    # Simulate delivery vs demand (r_t - q_t)
    r_minus_q = np.random.uniform(0, 100, N)
    
    def simulate_memory(alpha, rho, psi):
        m = np.zeros(N)
        g_alpha, b_coeff, theta = get_l1_coefficients(alpha, N, delta_tau, rho)
        
        for t in range(1, N):
            F_t = psi * r_minus_q[t]
            if alpha == 1.0:
                m[t] = theta * (m[t-1] + delta_tau * F_t)
            else:
                past_sum = sum(b_coeff[t - ell] * m[ell] for ell in range(1, t))
                m[t] = (g_alpha * F_t + past_sum) / (1 + g_alpha * rho)
        return m
        
    # True memory trajectory
    m_true = simulate_memory(alpha_true, rho_true, psi_true)
    
    # Add noise to simulate observation
    m_obs = m_true + np.random.normal(0, 5, N)
    
    # Objective function for calibration
    def loss_function(params):
        alpha, rho, psi = params
        if not (0.1 <= alpha <= 1.0 and rho >= 0 and psi >= 0):
            return 1e9
        m_pred = simulate_memory(alpha, rho, psi)
        return np.mean((m_pred - m_obs)**2)
        
    # Initial guess
    x0 = [0.5, 0.5, 1.0]
    res = scipy.optimize.minimize(loss_function, x0, method='L-BFGS-B', bounds=[(0.1, 1.0), (0.0, 1.0), (0.0, 2.0)])
    
    alpha_est, rho_est, psi_est = res.x
    print(f"Target: alpha={alpha_true}, rho={rho_true}, psi={psi_true}")
    print(f"Estimated: alpha={alpha_est:.3f}, rho={rho_est:.3f}, psi={psi_est:.3f}")
    # Plot 1: Trajectory
    plt.figure(figsize=(6, 4))
    plt.plot(m_true, label='True ($\\alpha=0.8$)', linestyle='--', color='black', linewidth=2)
    plt.plot(m_obs, label='Observed (Noisy)', marker='o', alpha=0.5, color='gray')
    m_pred = simulate_memory(alpha_est, rho_est, psi_est)
    plt.plot(m_pred, label='Estimated', color='blue', linewidth=2)
    plt.xlabel('Time Step $t$')
    plt.ylabel('Congestion Memory $m_t$')
    plt.title('Fractional Memory Identification')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('manuscript/figures/fig_calibration_trajectory.pdf')
    plt.close()
    
    # Plot 2: Surface
    alphas = np.linspace(0.5, 1.0, 20)
    rhos = np.linspace(0.0, 0.4, 20)
    A, R = np.meshgrid(alphas, rhos)
    Z = np.zeros_like(A)
    for i in range(20):
        for j in range(20):
            Z[i, j] = loss_function([A[i, j], R[i, j], psi_true])
    
    plt.figure(figsize=(6, 4))
    cp = plt.contourf(A, R, np.log10(Z + 1), levels=20, cmap='viridis')
    plt.colorbar(cp, label='$\\log_{10}(\\mathrm{MSE} + 1)$')
    plt.plot(alpha_true, rho_true, 'r*', markersize=15, label='True')
    plt.plot(alpha_est, rho_est, 'wo', markersize=8, markeredgecolor='k', label='Estimated')
    plt.xlabel('Fractional order $\\alpha$')
    plt.ylabel('Dissipation $\\rho$')
    plt.title('Loss slice $L(\\alpha,\\rho,\\psi^\\star)$ at $\\psi^\\star=0.5$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('manuscript/figures/fig_calibration_surface.pdf')
    plt.close()

    return alpha_est, rho_est, psi_est

def experiment_2_baselines():
    print("--- Experiment 2: Memory Baselines (DE) ---")
    tree = generate_toy_tree(N=2)
    
    scenarios = [
        {"name": "Zero memory forcing (psi=0)", "chi": 0.0, "alpha": 1.0, "psi": 0.0},
        {"name": "First-order (alpha=1, chi=0.5)", "chi": 0.5, "alpha": 1.0, "psi": 0.5},
        {"name": "Fractional (alpha=0.8, chi=0.5)", "chi": 0.5, "alpha": 0.8, "psi": 0.5},
    ]
    
    results = {}
    
    # Simulate step-response for plot
    N_sim = 15
    delta_tau = 1.0
    rho_sim = 0.1
    psi_sim = 1.0
    F_t = np.zeros(N_sim)
    F_t[3:8] = 50.0  # congestion spike
    
    m_alpha1 = np.zeros(N_sim)
    m_alpha08 = np.zeros(N_sim)
    
    g_1, _, theta_1 = get_l1_coefficients(1.0, N_sim, delta_tau, rho_sim)
    g_08, b_08, _ = get_l1_coefficients(0.8, N_sim, delta_tau, rho_sim)
    
    for t in range(1, N_sim):
        m_alpha1[t] = theta_1 * (m_alpha1[t-1] + delta_tau * F_t[t])
        
        past_sum = sum(b_08[t - ell] * m_alpha08[ell] for ell in range(1, t))
        m_alpha08[t] = (g_08 * F_t[t] + past_sum) / (1 + g_08 * rho_sim)
        
    plt.figure(figsize=(6, 4))
    plt.plot(F_t, label='Forcing $F_t$ (Congestion)', color='red', linestyle=':')
    plt.plot(np.zeros(N_sim), label='Zero memory forcing ($\\psi=0, m_0=0$)', color='gray')
    plt.plot(m_alpha1, label='First-order ($\\alpha=1$)', color='green', linestyle='--')
    plt.plot(m_alpha08, label='Fractional ($\\alpha=0.8$)', color='blue')
    plt.xlabel('Time Step $t$')
    plt.ylabel('Memory State $m_t$')
    plt.title('Memory Dynamics Response')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('manuscript/figures/fig_memory_baselines.pdf')
    plt.close()
    
    for scen in scenarios:
        params = TransportationModelParams()
        params.chi = {j: scen["chi"] for j in params.J}
        params.alpha = scen["alpha"]
        params.psi = {j: scen["psi"] for j in params.J}
        
        model, y, x, r, q, u, z, m, b, s, eta, xi = build_deterministic_equivalent(tree, params)
        model.solve(pulp.GUROBI(msg=False))
        
        if model.status == pulp.LpStatusOptimal:
            cost = pulp.value(model.objective)
            print(f"{scen['name']}: Optimal Cost = {cost:.2f}")
            results[scen['name']] = cost
        else:
            print(f"{scen['name']}: Solver failed. Status={pulp.LpStatus[model.status]}")
            
    print()
    return results

def experiment_3_ph():
    print("--- Experiment 3: Progressive Hedging vs DE ---")
    tree = generate_toy_tree(N=2)
    params = TransportationModelParams()
    
    # 1. Solve True DE
    model_de, y_de, x_de, r_de, q_de, u_de, z_de, m_de, b_de, s_de, eta_de, xi_de = build_deterministic_equivalent(tree, params)
    model_de.solve(pulp.GUROBI(msg=False))
    
    if model_de.status != pulp.LpStatusOptimal:
        print("DE failed to solve optimally.")
        return
        
    true_cost = pulp.value(model_de.objective)
    true_y = {k: pulp.value(y_de[k]) for k in params.K}
    print(f"True DE Cost: {true_cost:.2f}")
    print(f"True y: {true_y}")
    
    # 2. Run PH
    print("Running PH Algorithm...")
    y_hat, eta_bar, history = run_progressive_hedging(tree, params, rho_PH=500.0, epsilon=1e-2, max_iter=20)
    
    print(f"PH Converged in {len(history)} iterations.")
    print(f"PH Recovered y: {y_hat}")
    
    # 3. Evaluate PH Policy
    model_eval, y_eval, *_ = build_deterministic_equivalent(tree, params)
    # Fix y variables
    for k in params.K:
        y_eval[k].setInitialValue(y_hat[k])
        y_eval[k].fixValue()
        
    model_eval.solve(pulp.GUROBI(msg=False))
    ph_cost = pulp.value(model_eval.objective)
    
    if ph_cost is None:
        print("PH Policy is infeasible.")
    else:
        print(f"PH Policy Cost: {ph_cost:.2f}")
        optimality_gap = (ph_cost - true_cost) / true_cost * 100 if true_cost > 0 else 0
        print(f"Optimality Gap: {optimality_gap:.2f}%\n")
    
    # Save CSV
    import csv
    with open('computational_validation/results/ph_history.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['iteration', 'r_nac', 'r_proxy', 'stopping_reason'])
        writer.writeheader()
        for h in history:
            writer.writerow({
                'iteration': h['iteration'],
                'r_nac': h['r_nac'],
                'r_proxy': h['r_proxy'],
                'stopping_reason': h.get('stopping_reason', '')
            })
            
    # Plot PH Convergence
    iters = [h['iteration'] for h in history]
    rnac = [h['r_nac'] for h in history]
    
    plt.figure(figsize=(6, 4))
    plt.plot(iters, rnac, marker='s', color='purple', linewidth=2)
    plt.axhline(1e-2, color='k', linestyle='--', label='Tolerance $\\varepsilon=10^{-2}$')
    
    final_reason = history[-1].get('stopping_reason', 'maximum iterations')
    plt.annotate(f"Stopped: {final_reason}\\nFinal $R_{{{'NAC'}}} = {rnac[-1]:.4f}$",
                 xy=(iters[-1], rnac[-1]), xytext=(max(0, iters[-1]-5), rnac[-1]*2.0),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                 
    plt.xlabel('PH Iteration $r$')
    plt.ylabel('Non-Anticipativity Residual $R_{\\mathrm{NAC}}$')
    plt.yscale('log')
    plt.title('PWL-PH residual history for one illustrative run')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('manuscript/figures/fig_ph_convergence.pdf')
    plt.close()
    
    # Extract costs and carbon for True DE
    scen_costs = []
    scen_net_carbon = []
    for omega_dict in tree.scenarios:
        omega = omega_dict['omega']
        # Cost along scenario
        path = omega_dict['path']
        cost_omega = 0
        net_carbon = 0
        for node in path:
            n = node.node_id
            op_cost = sum(params.v[i, j, k] * x_de[i, j, k, n].varValue for i in params.I for j in params.J for k in params.K)
            inv_cost = sum(params.h[j] * z_de[j, n].varValue for j in params.J)
            pen_cost = sum(params.pi[j] * u_de[j, n].varValue for j in params.J)
            cost_omega += op_cost + inv_cost + pen_cost
            net_carbon += b_de[n].varValue - s_de[n].varValue
        scen_costs.append(cost_omega)
        scen_net_carbon.append(net_carbon)
        
    # Plot Scenario Costs
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(scen_costs)), scen_costs, color='steelblue')
    plt.xlabel('Scenario $\\omega$')
    plt.ylabel('Operational Cost $C^\\omega$')
    plt.title('Scenario Cost Distribution')
    plt.xticks(range(len(scen_costs)))
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('manuscript/figures/fig_scenario_costs.pdf')
    plt.close()
    
    # Plot Carbon Trading
    plt.figure(figsize=(6, 4))
    bars = plt.bar(range(len(scen_net_carbon)), scen_net_carbon, color='forestgreen')
    for idx, val in enumerate(scen_net_carbon):
        if val < 0:
            bars[idx].set_color('crimson')
    plt.axhline(0, color='k', linewidth=1)
    plt.xlabel('Scenario $\\omega$')
    plt.ylabel('Net Allowances (Bought - Sold)')
    plt.title('Carbon Trading Position per Scenario')
    plt.xticks(range(len(scen_net_carbon)))
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('manuscript/figures/fig_carbon_trading.pdf')
    plt.close()

if __name__ == "__main__":
    experiment_1_calibration()
    experiment_2_baselines()
    experiment_3_ph()
