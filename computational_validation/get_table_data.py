import sys
sys.path.append('src')
from progressive_hedging import run_progressive_hedging
from model_builder import build_deterministic_equivalent, TransportationModelParams
from scenario_tree import generate_toy_tree
import pulp

tree = generate_toy_tree()
params = TransportationModelParams()

model_de, y_de, x_de, r_de, q_de, u_de, z_de, m_de, b_de, s_de, eta_de, xi_de = build_deterministic_equivalent(tree, params)
model_de.solve(pulp.GUROBI(msg=False))

print("\\begin{table}[ht]")
print("\\centering")
print("\\caption{Scenario-level breakdowns for the toy instance. All variables satisfy $b_t^\\omega s_t^\\omega = 0$.}")
print("\\begin{tabular}{lrrrrrrrr}")
print("\\toprule")
print("Scen & Prob & Op.Cost & Lost-Sales & Holding & Net-Carbon & Emissions & Buy & Sell \\\\")
print("\\midrule")
for i, sc in enumerate(tree.scenarios):
    omega = sc['omega']
    prob = sc['prob']
    op_cost = sum(params.v[i, j, k] * x_de[i, j, k, omega].varValue for i in params.I for j in params.J for k in params.K)
    ls_cost = sum(params.pi[j] * u_de[j, omega].varValue for j in params.J)
    holding = sum(params.h[j] * z_de[j, omega].varValue for j in params.J)
    emissions = sum(params.e[i, j, k] * x_de[i, j, k, omega].varValue for i in params.I for j in params.J for k in params.K)
    purchases = b_de[omega].varValue
    sales = s_de[omega].varValue
    carbon_net = params.P_B * purchases - params.P_S * sales
    
    print(f"$\\omega_{omega}$ & {prob:.2f} & {op_cost:.1f} & {ls_cost:.1f} & {holding:.1f} & {carbon_net:.1f} & {emissions:.1f} & {purchases:.1f} & {sales:.1f} \\\\")
    
print("\\bottomrule")
print("\\end{tabular}")
print("\\label{tab:scenarios}")
print("\\end{table}")
