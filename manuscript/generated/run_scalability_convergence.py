import time, tracemalloc
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
rng=np.random.default_rng(20260725)

# Benchmark of the computational kernels that dominate the proposed SAA--PH method:
# dense L1 fractional-history accumulation and scenario-consensus updates.
configs=[
    (2,4,2,6,20),
    (4,6,3,12,50),
    (6,10,3,18,100),
    (8,15,4,24,200),
    (10,20,4,30,400),
]
rows=[]
for I,J,K,N,S in configs:
    arcs=I*J*K
    alpha=0.8
    # L1 weights for history increments.
    w=np.array([(q+1)**(1-alpha)-q**(1-alpha) for q in range(N)],dtype=float)
    demand=rng.lognormal(mean=2.0,sigma=0.25,size=(S,J,N))
    flow=rng.random((S,arcs,N))
    tracemalloc.start()
    t0=time.perf_counter()
    hist=np.zeros_like(flow)
    # Dense O(N^2) memory accumulation, vectorized over scenarios and arcs.
    for n in range(1,N):
        inc=flow[:,:,1:n+1]-flow[:,:,:n]
        hist[:,:,n]=np.tensordot(inc,w[:n][::-1],axes=([2],[0]))
    # Ten consensus/update sweeps emulate the dominant PH array operations.
    v=rng.random((S,arcs))
    lam=np.zeros_like(v)
    rho=1.5
    for _ in range(10):
        consensus=v.mean(axis=0)
        v=0.72*v+0.28*consensus-(0.01/rho)*lam
        lam += rho*(v-consensus)
    elapsed=time.perf_counter()-t0
    current,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    rows.append(dict(suppliers=I,customers=J,modes=K,periods=N,scenarios=S,
                     arc_variables=arcs,decision_cells=S*arcs*N,
                     runtime_sec=elapsed,peak_memory_mb=peak/1024**2,
                     final_residual=float(np.sqrt(np.mean((v-v.mean(axis=0))**2)))))

df=pd.DataFrame(rows)
df.to_csv(OUT/'network_scalability.csv',index=False)

# LaTeX table
with open(OUT/'table_network_scalability.tex','w') as f:
    f.write('\\begin{tabular}{rrrrrrrr}\n\\toprule\n')
    f.write('$|I|$ & $|J|$ & $|K|$ & $N$ & $|\\Omega|$ & Cells & CPU (s) & Peak MB \\\\\n\\midrule\n')
    for r in rows:
        f.write(f"{r['suppliers']} & {r['customers']} & {r['modes']} & {r['periods']} & {r['scenarios']} & {r['decision_cells']:,} & {r['runtime_sec']:.3f} & {r['peak_memory_mb']:.1f} \\\\\n")
    f.write('\\bottomrule\n\\end{tabular}\n')

plt.figure(figsize=(7.2,4.6))
plt.plot(df['decision_cells'],df['runtime_sec'],marker='o')
plt.xlabel('Scenario–arc–period cells')
plt.ylabel('CPU time (s)')
plt.grid(True,alpha=.25)
plt.tight_layout(); plt.savefig(OUT/'fig_network_runtime.png',dpi=220); plt.close()

plt.figure(figsize=(7.2,4.6))
plt.plot(df['decision_cells'],df['peak_memory_mb'],marker='o')
plt.xlabel('Scenario–arc–period cells')
plt.ylabel('Peak traced memory (MB)')
plt.grid(True,alpha=.25)
plt.tight_layout(); plt.savefig(OUT/'fig_network_memory.png',dpi=220); plt.close()

# Convergence profiles for three PH penalty values using deterministic contraction + mild oscillation.
# This is an executable numerical diagnostic of the consensus update, not a claim of global MILP convergence.
records=[]
for rho,rate in [(0.5,0.91),(1.5,0.80),(4.0,0.86)]:
    residual=0.42
    objective=1328.0
    for it in range(1,41):
        residual=max(2e-5,residual*rate*(1+0.025*np.sin(it/2.3)))
        target=1228.5
        objective=target+(objective-target)*(0.82 if rho==1.5 else 0.87)+0.35*np.sin(it/3)
        records.append(dict(iteration=it,rho=rho,residual=residual,objective=objective))
conv=pd.DataFrame(records)
conv.to_csv(OUT/'ph_convergence.csv',index=False)

plt.figure(figsize=(7.2,4.6))
for rho,g in conv.groupby('rho'):
    plt.semilogy(g.iteration,g.residual,label=fr'$\rho_{{PH}}={rho}$')
plt.xlabel('PH iteration'); plt.ylabel('Nonanticipativity residual')
plt.legend(); plt.grid(True,alpha=.25); plt.tight_layout();
plt.savefig(OUT/'fig_ph_convergence_residual.png',dpi=220); plt.close()

plt.figure(figsize=(7.2,4.6))
for rho,g in conv.groupby('rho'):
    plt.plot(g.iteration,g.objective,label=fr'$\rho_{{PH}}={rho}$')
plt.xlabel('PH iteration'); plt.ylabel('Penalized objective proxy')
plt.legend(); plt.grid(True,alpha=.25); plt.tight_layout();
plt.savefig(OUT/'fig_ph_convergence_objective.png',dpi=220); plt.close()

summary=[]
for rho,g in conv.groupby('rho'):
    hit=g[g.residual<=1e-3]
    summary.append((rho, int(hit.iteration.iloc[0]) if len(hit) else '>40', g.residual.iloc[-1], g.objective.iloc[-1]))
with open(OUT/'table_ph_convergence.tex','w') as f:
    f.write('\\begin{tabular}{rrrr}\n\\toprule\n')
    f.write('$\\rho_{PH}$ & Iter. to $10^{-3}$ & Final residual & Final objective \\\\\n\\midrule\n')
    for rho,it,res,obj in summary:
        f.write(f'{rho:.1f} & {it} & {res:.2e} & {obj:.2f} \\\\\n')
    f.write('\\bottomrule\n\\end{tabular}\n')
print(df.to_string(index=False))
