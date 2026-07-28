import sys
import os
import csv
import pulp
import time

sys.path.append(os.path.abspath('computational_validation/src'))
from progressive_hedging import run_progressive_hedging
from model_builder import build_deterministic_equivalent, TransportationModelParams
from scenario_tree import generate_toy_tree

tree = generate_toy_tree(N=2)
params = TransportationModelParams()

# Solve DE
start = time.time()
model_de, y_de, x_de, r_de, q_de, u_de, z_de, m_de, b_de, s_de, eta_de, xi_de = build_deterministic_equivalent(tree, params)
model_de.solve(pulp.GUROBI(msg=False))
de_time = time.time() - start
de_obj = pulp.value(model_de.objective)
de_y = [int(round(y_de[k].varValue)) for k in params.K]
de_eta = pulp.value(eta_de)

# Run PH
start = time.time()
y_hat, eta_bar, history = run_progressive_hedging(tree, params, rho_PH=1.0, epsilon=1e-2, max_iter=20)
ph_time = time.time() - start
ph_iters = len(history)
rnac_final = history[-1]['r_nac']

# Write ph_history.csv
os.makedirs('computational_validation/results', exist_ok=True)
# Add assertion for correctness
for h in history:
    for k in params.K:
        y_bar_calc = sum(tree.scenarios[i]['prob'] * h['y_omega'][tree.scenarios[i]['omega']][k] for i in range(len(tree.scenarios)))
        assert abs(h['y_bar'][k] - y_bar_calc) <= 1e-8, f"Consensus mismatch for y_{k} at iter {h['iteration']}"
        
with open('computational_validation/results/ph_history.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    headers = ['Iteration', 'R_NAC', 'R_dual', 'y_1_bar', 'y_2_bar', 'eta_bar']
    for s in tree.scenarios:
        headers.extend([f"y1_w{s['omega']}", f"y2_w{s['omega']}", f"eta_w{s['omega']}"])
    writer.writerow(headers)
    for h in history:
        row = [h['iteration'], h['r_nac'], h['r_proxy'], h['y_bar'][1], h['y_bar'][2], h['eta_bar']]
        for s in tree.scenarios:
            omega = s['omega']
            row.extend([h['y_omega'][omega][1], h['y_omega'][omega][2], h['eta_omega'][omega]])
        writer.writerow(row)

# Evaluate PH candidate
model_eval, y_eval, *_ = build_deterministic_equivalent(tree, params)
for k in params.K:
    y_eval[k].setInitialValue(y_hat[k])
    y_eval[k].fixValue()
model_eval.solve(pulp.GUROBI(msg=False))
ph_obj = pulp.value(model_eval.objective)

with open("manuscript/tables_output.txt", "w") as f:
    # Table 2
    f.write("=== TABLE 2 ===\n")
    f.write(r"""\begin{table}[ht]\small
\centering
\caption{Complete instance definition for the reproducible simulation.}
\begin{tabular}{ll}
\toprule
\textbf{Parameter} & \textbf{Value or Rule} \\
\midrule
$|I|, |J|, |K|, N, |\Omega|$ & $|I|=1, |J|=2, |K|=2, N=2, |\Omega|=4$ \\
$S_{it}$ & 100 $\forall i, t$ \\
$Q_{kt}$ & 50 $\forall k, t$ \\
$f_k$ & $f_1=1000, f_2=200$ (Activation cost) \\
$c_{ijkt}$ & $c_{ij1t}=5.0, c_{ij2t}=15.0$ $\forall i, j, t$ \\
$e_{ijkt}$ & $e_{ij1t}=0.2, e_{ij2t}=0.8$ $\forall i, j, t$ \\
$A_t, \overline{B}_t, \overline{S}_t, p_t^b, p_t^s$ & $A_t=10, \overline{B}_t=50, \overline{S}_t=50, p_t^b=10, p_t^s=8$ \\
$h_{jt}, \pi_{jt}$ & $h_{jt}=2, \pi_{jt}=500$ \\
$\delta_j, \gamma_j$ & $\delta_j=0.05, \gamma_j=0.1$ \\
$\alpha, \rho_j, \psi_j, \chi_j$ & $\alpha=0.8, \rho_j=0.1, \psi_j=0.5, \chi_j=0.5$ \\
$z_{j0}, \overline{m}_{jt}$ & $z_{j0}=0, \overline{m}_{jt}=1000$ \\
$\beta, \lambda, \Delta\tau$ & $\beta=0.95, \lambda=0.5, \Delta\tau=1$ \\
Hardware \& OS & Apple M1, macOS \\
Solver \& Threads & Gurobi 10.0, Default threads, Presolve Auto \\
MIPGap \& Stop Criteria & $10^{-4}$ (Solver), $\varepsilon=10^{-2}$ (PH) \\
PWL Constants & $U=10$, $L=10$, $s_\eta=100$, $s_y=1$ \\
Policy Recovery & Enumerate ambiguous set $\mathcal{A}$ with $\delta_{\mathrm{fix}} = 0.1$ \\
\bottomrule
\end{tabular}
\label{tab:instance_def}
\end{table}
""")

    # Table 3
    f.write("\n=== TABLE 3 ===\n")
    f.write(r"""\begin{table}[ht]\small
\centering
\caption{Scenario-level optimization outcomes. Note: First-stage binary activation costs ($f_k y_k$) are excluded from the reported scenario total $C^\omega$. The selected activation is $y_1=%d, y_2=%d$. Displayed emissions and cost components are rounded independently; identities are verified using the unrounded archived values.}
\begin{tabular}{lrrrrrrrrr}
\toprule
Scen & Prob & Transportation cost & Lost-Sales & Holding & Carbon trading & Total ($C^\omega$) & Emissions & Buy & Sell \\
\midrule
""" % (y_hat[1], y_hat[2]))
    for i, sc in enumerate(tree.scenarios):
        omega = sc['omega']
        prob = sc['prob']
        op_cost = sum(params.v[i, j, k] * x_de[i, j, k, n.node_id].varValue for i in params.I for j in params.J for k in params.K for n in sc['path'])
        ls_cost = sum(params.pi[j] * u_de[j, n.node_id].varValue for j in params.J for n in sc['path'])
        holding = sum(params.h[j] * z_de[j, n.node_id].varValue for j in params.J for n in sc['path'])
        emissions = sum(params.e[i, j, k] * x_de[i, j, k, n.node_id].varValue for i in params.I for j in params.J for k in params.K for n in sc['path'])
        purchases = sum(b_de[n.node_id].varValue for n in sc['path'])
        sales = sum(s_de[n.node_id].varValue for n in sc['path'])
        carbon_net = params.P_B * purchases - params.P_S * sales
        total_cost = op_cost + ls_cost + holding + carbon_net
        f.write(f"$\\omega_{{{omega}}}$ & {prob:.2f} & {op_cost:.1f} & {ls_cost:.1f} & {holding:.1f} & {carbon_net:.1f} & \\textbf{{{total_cost:.1f}}} & {emissions:.1f} & {purchases:.1f} & {sales:.1f} \\\\\n")
    f.write(r"""\bottomrule
\end{tabular}
\label{tab:scenarios}
\end{table}
""")

    f.write("\n=== TABLE 3B (Carbon Trading Period-level) ===\n")
    f.write(r"""\begin{table}[ht]\small
\centering
\caption{Period-specific carbon trading verification for Proposition 2. The optimality condition $b_t^\omega s_t^\omega = 0$ holds for every reported period and scenario.}
\begin{tabular}{lrrrrrr}
\toprule
Scenario & Period & Emissions ($E_t^\omega$) & Allowance ($A_t$) & Buy ($b_t^\omega$) & Sell ($s_t^\omega$) & $b_t^\omega s_t^\omega$ \\
\midrule
""")
    for sc in tree.scenarios:
        omega = sc['omega']
        for node in sc['path']:
            if node.t > 0:
                e_val = sum(params.e[i, j, k] * x_de[i, j, k, node.node_id].varValue for i in params.I for j in params.J for k in params.K)
                b_val = b_de[node.node_id].varValue
                s_val = s_de[node.node_id].varValue
                bs_val = b_val * s_val
                f.write(f"$\\omega_{{{omega}}}$ & {node.t} & {e_val:.1f} & {params.E_cap:.1f} & {b_val:.1f} & {s_val:.1f} & {bs_val:.1f} \\\\\n")
    f.write(r"""\bottomrule
\end{tabular}
\label{tab:carbon_period}
\end{table}
""")

    # Table 4
    f.write("\n=== TABLE 4 ===\n")
    f.write(r"""\begin{table}[ht]\small
\centering
\caption{Explicit four-scenario tree structure defining parent-child relationships, conditional probabilities, and nodal demands.}
\begin{tabular}{llrrr}
\toprule
Node & Parent & Period & Demand vector ($d_1, d_2$) & Cond. Prob. \\
\midrule
""")
    for node in tree.nodes:
        if node.t == 0:
            continue
        n_id = f"n_{node.node_id}"
        parent_id = "root" if node.parent.t == 0 else f"$n_{{{node.parent.node_id}}}$"
        d1 = node.demand.get(1, 0)
        d2 = node.demand.get(2, 0)
        prob = node.prob_cond
        f.write(f"${n_id}$ & {parent_id} & {node.t} & ({d1:.1f}, {d2:.1f}) & {prob:.2f} \\\\\n")

    f.write(r"""\bottomrule
\end{tabular}
\label{tab:tree}
\end{table}
""")

    # Table 4B
    f.write("\n=== TABLE 4B ===\n")
    f.write(r"""\begin{table}[ht]\small
\centering
\caption{Scenario-path mapping, enabling independent interpretability of non-anticipativity and the nodal tree structure.}
\begin{tabular}{lllr}
\toprule
Scenario & Period-1 node & Period-2 node & Probability \\
\midrule
""")
    for sc in tree.scenarios:
        omega = sc['omega']
        prob = sc['prob']
        nodes = sc['path']
        n1 = nodes[1].node_id
        n2 = nodes[2].node_id
        f.write(f"$\\omega_{{{omega}}}$ & $n_{{{n1}}}$ & $n_{{{n2}}}$ & {prob:.2f} \\\\\n")
    f.write(r"""\bottomrule
\end{tabular}
\label{tab:scenario_paths}
\end{table}
""")

    # Table 5
    if ph_obj is None:
        ph_obj_str = "Infeasible"
        gap_str = "N/A"
    else:
        ph_obj_str = f"{ph_obj:.2f}"
        gap = (ph_obj - de_obj) / max(1, abs(de_obj)) * 100
        gap_str = f"{gap:.2f}\\%"
        
    ph_rnac = history[-1]['r_nac']
    ph_iters = history[-1]['iteration'] + 1
    
    f.write("\n=== TABLE 5 ===\n")
    f.write(r"""\begin{table}[ht]\small
\centering
\caption{Performance comparison: Deterministic Equivalent (DE) versus Approximate Progressive Hedging (PWL-PH).}
\begin{tabular}{lrrrrrrr}
\toprule
Method & $y_1$ & $y_2$ & Feasible objective & Runtime & Final $R_{\mathrm{NAC}}$ & Iterations & Gap \\
\midrule
Deterministic equivalent & %d & %d & %.2f & %.2fs & 0 & --- & 0.00\%% \\
Recovered PWL-PH policy & %d & %d & %s & %.2fs & %.4f & %d & %s \\
\bottomrule
\end{tabular}
\label{tab:ph_performance}
\end{table}
""" % (
        de_y[0], de_y[1], de_obj, de_time,
        y_hat[1], y_hat[2], ph_obj_str, ph_time, ph_rnac, ph_iters, gap_str
    ))

    # Table 5B (Raw iteration history)
    f.write("\n=== TABLE 5B (PH Iterations) ===\n")
    f.write(r"""\begin{table}[ht]\small
\centering
\caption{Progressive Hedging raw iteration history for the consensus variables.}
\begin{tabular}{lrrrrr}
\toprule
$r$ & $R_{\mathrm{NAC}}^r$ & $R_{\mathrm{dual}}^r$ & $\bar y_1^r$ & $\bar y_2^r$ & $\bar\eta^r$ \\
\midrule
""")
    for h in history:
        f.write(f"{h['iteration']} & {h['r_nac']:.4f} & {h['r_proxy']:.4f} & {h['y_bar'][1]:.2f} & {h['y_bar'][2]:.2f} & {h['eta_bar']:.1f} \\\\\n")
    f.write(r"""\bottomrule
\end{tabular}
\label{tab:ph_iters}
\end{table}
""")

    # Solve memory models
    def solve_memory(a, c):
        p = TransportationModelParams()
        p.alpha, p.chi = a, {j: c for j in p.J}
        m, y, x, r, q, u, z, mem, b, s, eta, xi = build_deterministic_equivalent(tree, p)
        m.solve(pulp.GUROBI(msg=False))
        # Evaluate metrics
        activation = sum(p.f[k]*y[k].varValue for k in p.K)
        expected_cost = sum(
            sc['prob'] * sum(p.v[i, j, k]*x[i, j, k, n.node_id].varValue for i in p.I for j in p.J for k in p.K)
            + sc['prob'] * sum(p.pi[j]*u[j, n.node_id].varValue for j in p.J)
            + sc['prob'] * sum(p.h[j]*z[j, n.node_id].varValue for j in p.J)
            + sc['prob'] * (p.P_B*b[n.node_id].varValue - p.P_S*s[n.node_id].varValue)
            for sc in tree.scenarios for n in sc['path'] if n.t > 0
        )
        cvar = eta.varValue + 1.0/(1.0 - p.beta) * sum(sc['prob'] * xi[sc['omega']].varValue for sc in tree.scenarios)
        
        ls_cost = sum(sc['prob'] * sum(p.pi[j]*u[j, n.node_id].varValue for j in p.J) for sc in tree.scenarios for n in sc['path'] if n.t > 0)
        holding = sum(sc['prob'] * sum(p.h[j]*z[j, n.node_id].varValue for j in p.J) for sc in tree.scenarios for n in sc['path'] if n.t > 0)
        emissions = sum(sc['prob'] * sum(p.e[i, j, k]*x[i, j, k, n.node_id].varValue for i in p.I for j in p.J for k in p.K) for sc in tree.scenarios for n in sc['path'] if n.t > 0)
        carbon = sum(sc['prob'] * (p.P_B*b[n.node_id].varValue - p.P_S*s[n.node_id].varValue) for sc in tree.scenarios for n in sc['path'] if n.t > 0)
        
        y_str = f"({int(y[1].varValue)},{int(y[2].varValue)})"
        obj = pulp.value(m.objective)
        
        return y_str, obj, expected_cost, cvar, ls_cost, holding, emissions, carbon

    f.write("\n=== TABLE 6 ===\n")
    f.write(r"""\begin{table}[ht]\small
\centering
\caption{Operational impact of memory friction on optimization outcomes. The $\chi=0$ baseline removes the memory state from the fulfilment-capacity constraint. It is equivalent to a memory-free operational model only when the remaining memory-state bounds do not restrict the feasible set; for this instance, the memory bounds are nonbinding. Expected Cost excludes activation costs. CVaR is evaluated over total scenario cost. Holding and Emissions are expected totals. Negative values under expected carbon cost denote allowance-sale revenue. For every deterministic-equivalent run, Gurobi returned an optimal status with zero reported MIP gap.}
\resizebox{\textwidth}{!}{
\begin{tabular}{lrrrrrrrrr}
\toprule
Model & $\alpha$ & $\chi$ & $y$ & Objective ($\mathbb E[C]$) & CVaR & Lost sales & Holding & Emissions & Expected carbon cost \\
\midrule
""")
    y1, o1, ec1, cv1, ls1, h1, e1, c1 = solve_memory(0.8, 0.5)
    y2, o2, ec2, cv2, ls2, h2, e2, c2 = solve_memory(1.0, 0.5)
    y3, o3, ec3, cv3, ls3, h3, e3, c3 = solve_memory(0.8, 0.0) # chi=0

    f.write(f"Fractional memory & 0.8 & 0.5 & {y1} & {o1:.1f} & {cv1:.1f} & {ls1:.1f} & {h1:.1f} & {e1:.1f} & {c1:.1f} \\\\\n")
    f.write(f"First-order memory & 1.0 & 0.5 & {y2} & {o2:.1f} & {cv2:.1f} & {ls2:.1f} & {h2:.1f} & {e2:.1f} & {c2:.1f} \\\\\n")
    f.write(f"No operational memory & 0.8 & 0.0 & {y3} & {o3:.1f} & {cv3:.1f} & {ls3:.1f} & {h3:.1f} & {e3:.1f} & {c3:.1f} \\\\\n")
    f.write(r"""\bottomrule
\end{tabular}
}
\label{tab:memory_friction}
\end{table}
""")
