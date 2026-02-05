# high_impact_tests.py
# 4 Critical Tests for High-Impact Decision

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import sys
sys.path.insert(0, '/home/runner/workspace/src')
from rpm_shelves import singlet_yield, anisotropy, _compute_anisotropy_point
import time

def singlet_yield_at_theta(B_uT, theta, A=1.0, J=0.5, kS=1.0, kT=0.1, dephase=0.02, tmax=6.0, dt=0.03):
    """Compute singlet yield at specific angle theta."""
    return singlet_yield(B_uT=B_uT, theta=theta, A=A, J=J, kS=kS, kT=kT, dephase=dephase, tmax=tmax, dt=dt)

def anisotropy_custom_theta(B_uT, theta_ref=0.0, theta_compare=np.pi/4, **kwargs):
    """Compute anisotropy between two arbitrary angles."""
    y1 = singlet_yield(B_uT=B_uT, theta=theta_ref, **kwargs)
    y2 = singlet_yield(B_uT=B_uT, theta=theta_compare, **kwargs)
    return y1 - y2

# ============================================================
# RUN 1: Heatmap at θ=π/4
# ============================================================
def run_theta45_heatmap(grid_size=31, J=0.5, tmax=4.0, dt=0.05):
    """Heatmap comparing θ=0 vs θ=π/4 instead of θ=0 vs θ=π/2."""
    print(f"Running θ=π/4 heatmap (J={J}, {grid_size}x{grid_size})...")
    start = time.time()
    
    Bs = np.linspace(0, 100, grid_size)
    ratios = np.logspace(0, 3, grid_size)
    kT = 0.1
    
    Z = np.zeros((len(ratios), len(Bs)))
    
    for i, r in enumerate(ratios):
        kS = r * kT
        for j, B in enumerate(Bs):
            y0 = singlet_yield(B_uT=B, theta=0.0, A=1.0, J=J, kS=kS, kT=kT, dephase=0.02, tmax=tmax, dt=dt)
            y45 = singlet_yield(B_uT=B, theta=np.pi/4, A=1.0, J=J, kS=kS, kT=kT, dephase=0.02, tmax=tmax, dt=dt)
            Z[i,j] = y0 - y45
        if (i+1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(ratios)}")
    
    elapsed = time.time() - start
    print(f"Completed in {elapsed:.1f}s")
    
    suffix = "05" if J == 0.5 else "10"
    
    plt.figure(figsize=(10, 8))
    plt.imshow(Z, aspect="auto", origin="lower",
               extent=[Bs[0], Bs[-1], np.log10(ratios[0]), np.log10(ratios[-1])],
               cmap='viridis')
    plt.xlabel("B (µT)", fontsize=12)
    plt.ylabel("log₁₀(kS/kT)", fontsize=12)
    plt.title(f"Anisotropy (θ=0 vs θ=π/4), J={J}", fontsize=14)
    plt.colorbar(label="Δ singlet yield")
    plt.tight_layout()
    
    fname = f"../runs/heatmap_J{suffix}_theta45.png"
    plt.savefig(fname, dpi=200)
    print(f"Saved: {fname}")
    
    np.savez(f"../runs/heatmap_J{suffix}_theta45_data.npz", Z=Z, Bs=Bs, ratios=ratios, J=J)
    
    return Z, Bs, ratios

# ============================================================
# RUN 2: Ridge Extraction (B* vs log(kS/kT))
# ============================================================
def run_ridge_extraction(grid_size=41, tmax=4.0, dt=0.05):
    """Extract ridge: find B* where anisotropy is maximum for each kS/kT."""
    print("Running ridge extraction (diagonal law)...")
    start = time.time()
    
    Bs = np.linspace(5, 100, grid_size)  # Skip B=0
    ratios = np.logspace(0, 3, 31)
    kT = 0.1
    J = 0.5
    
    B_star = []
    max_aniso = []
    
    for r in ratios:
        kS = r * kT
        anisos = []
        for B in Bs:
            y0 = singlet_yield(B_uT=B, theta=0.0, A=1.0, J=J, kS=kS, kT=kT, dephase=0.02, tmax=tmax, dt=dt)
            y90 = singlet_yield(B_uT=B, theta=np.pi/2, A=1.0, J=J, kS=kS, kT=kT, dephase=0.02, tmax=tmax, dt=dt)
            anisos.append(y0 - y90)
        
        idx_max = np.argmax(anisos)
        B_star.append(Bs[idx_max])
        max_aniso.append(anisos[idx_max])
        print(f"  kS/kT={r:.1f}: B*={Bs[idx_max]:.1f}µT, max_aniso={anisos[idx_max]:.6f}")
    
    elapsed = time.time() - start
    print(f"Completed in {elapsed:.1f}s")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.semilogx(ratios, B_star, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel("kS/kT", fontsize=12)
    ax1.set_ylabel("B* (µT)", fontsize=12)
    ax1.set_title("Ridge: Optimal B vs kS/kT", fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    ax2.semilogx(ratios, max_aniso, 'r-o', linewidth=2, markersize=6)
    ax2.set_xlabel("kS/kT", fontsize=12)
    ax2.set_ylabel("Max anisotropy", fontsize=12)
    ax2.set_title("Peak anisotropy along ridge", fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle("Ridge Extraction: Diagonal Law", fontsize=16)
    plt.tight_layout()
    plt.savefig("../runs/ridge_Bstar_vs_logratio.png", dpi=200)
    print("Saved: ../runs/ridge_Bstar_vs_logratio.png")
    
    np.savez("../runs/ridge_data.npz", ratios=ratios, B_star=B_star, max_aniso=max_aniso)
    
    return ratios, B_star, max_aniso

# ============================================================
# RUN 3: Two-electron Hyperfine Symmetry Test
# ============================================================
def run_symmetry_test(n_points=21, tmax=4.0, dt=0.05):
    """Compare: one nucleus coupled to e1 vs both e1 and e2."""
    print("Running two-electron hyperfine symmetry test...")
    
    # This requires modifying the Hamiltonian
    # For now, we compare different A values as proxy
    
    from numpy import kron
    from scipy.linalg import expm
    
    sx = np.array([[0,1],[1,0]], dtype=complex)/2
    sy = np.array([[0,-1j],[1j,0]], dtype=complex)/2
    sz = np.array([[1,0],[0,-1]], dtype=complex)/2
    id2 = np.eye(2, dtype=complex)
    
    def op_e1(a): return kron(kron(a, id2), id2)
    def op_e2(a): return kron(kron(id2, a), id2)
    def op_n(a):  return kron(kron(id2, id2), a)
    
    S1x,S1y,S1z = op_e1(sx), op_e1(sy), op_e1(sz)
    S2x,S2y,S2z = op_e2(sx), op_e2(sy), op_e2(sz)
    Ix,Iy,Iz = op_n(sx), op_n(sy), op_n(sz)
    
    up = np.array([1,0],complex); dn = np.array([0,1],complex)
    S = (np.kron(up,dn) - np.kron(dn,up))/np.sqrt(2)
    P_S_e = np.outer(S,S.conj())
    P_T_e = np.eye(4, dtype=complex)-P_S_e
    P_S = kron(P_S_e, id2)
    P_T = kron(P_T_e, id2)
    rho0 = P_S / np.trace(P_S)
    
    def commutator_super(H):
        d = H.shape[0]; I = np.eye(d, dtype=complex)
        return -1j*(np.kron(H, I) - np.kron(I, H.T))
    
    def lindblad_super(L):
        d = L.shape[0]; I = np.eye(d, dtype=complex)
        LdL = L.conj().T @ L
        return np.kron(L, L.conj()) - 0.5*np.kron(LdL, I) - 0.5*np.kron(I, LdL.T)
    
    def singlet_yield_2e(B_uT, theta, A1, A2, J, kS, kT, dephase, tmax, dt):
        """Two-electron hyperfine: A1 on e1, A2 on e2."""
        d = 8
        omega = B_uT / 50.0
        Bx, Bz = omega*np.sin(theta), omega*np.cos(theta)
        
        HZ = Bx*(S1x+S2x) + Bz*(S1z+S2z)
        HHF1 = A1*(S1x@Ix + S1y@Iy + S1z@Iz)
        HHF2 = A2*(S2x@Ix + S2y@Iy + S2z@Iz)
        HJ = J*(S1x@S2x + S1y@S2y + S1z@S2z)
        H = HZ + HHF1 + HHF2 + HJ
        
        L = commutator_super(H)
        I = np.eye(d, dtype=complex)
        loss = -(kS/2)*(np.kron(P_S, I)+np.kron(I, P_S.T)) -(kT/2)*(np.kron(P_T, I)+np.kron(I, P_T.T))
        Ltot = L + loss
        if dephase > 0:
            Ltot += dephase*lindblad_super(S1z) + dephase*lindblad_super(S2z)
        
        U = expm(Ltot*dt)
        rho = rho0.copy()
        y = 0.0
        for _ in range(int(tmax/dt)):
            y += kS * float(np.real(np.trace(P_S @ rho))) * dt
            rho = (U @ rho.reshape(-1, order='F')).reshape((d,d), order='F')
        return y
    
    def aniso_2e(B, A1, A2, J=0.5, kS=1.0, kT=0.1, dephase=0.02):
        y0 = singlet_yield_2e(B, 0.0, A1, A2, J, kS, kT, dephase, tmax, dt)
        y90 = singlet_yield_2e(B, np.pi/2, A1, A2, J, kS, kT, dephase, tmax, dt)
        return y0 - y90
    
    Bs = np.linspace(0, 100, n_points)
    
    print("  Computing asymmetric (A1=1, A2=0)...")
    asym = [aniso_2e(B, 1.0, 0.0) for B in Bs]
    
    print("  Computing symmetric (A1=0.5, A2=0.5)...")
    sym = [aniso_2e(B, 0.5, 0.5) for B in Bs]
    
    print("  Computing enhanced (A1=1, A2=1)...")
    enh = [aniso_2e(B, 1.0, 1.0) for B in Bs]
    
    plt.figure(figsize=(10, 6))
    plt.plot(Bs, asym, 'b-', linewidth=2, label='Asymmetric (A1=1, A2=0)')
    plt.plot(Bs, sym, 'g--', linewidth=2, label='Symmetric (A1=0.5, A2=0.5)')
    plt.plot(Bs, enh, 'r:', linewidth=2.5, label='Enhanced (A1=1, A2=1)')
    plt.xlabel("B (µT)", fontsize=12)
    plt.ylabel("Anisotropy", fontsize=12)
    plt.title("Two-electron Hyperfine Symmetry Test", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../runs/symmetry_boost_comparison.png", dpi=200)
    print("Saved: ../runs/symmetry_boost_comparison.png")
    
    return Bs, asym, sym, enh

# ============================================================
# RUN 4: Angular Resolution Test
# ============================================================
def run_angular_resolution(n_angles_list=[8, 16, 32], B_test=50.0, tmax=4.0, dt=0.05):
    """Test angular resolution: compare 8/16/32 angle sampling."""
    print("Running angular resolution comparison...")
    
    kS, kT, J = 1.0, 0.1, 0.5
    
    results = {}
    
    for n_angles in n_angles_list:
        thetas = np.linspace(0, np.pi, n_angles)
        yields = [singlet_yield(B_uT=B_test, theta=th, A=1.0, J=J, kS=kS, kT=kT, 
                                dephase=0.02, tmax=tmax, dt=dt) for th in thetas]
        results[n_angles] = (thetas, yields)
        print(f"  n={n_angles}: range=[{min(yields):.6f}, {max(yields):.6f}]")
    
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red']
    for (n, (thetas, yields)), color in zip(results.items(), colors):
        plt.plot(np.degrees(thetas), yields, 'o-', color=color, linewidth=2, 
                 markersize=8 if n==8 else 5, label=f'{n} angles')
    
    plt.xlabel("θ (degrees)", fontsize=12)
    plt.ylabel("Singlet yield", fontsize=12)
    plt.title(f"Angular Resolution Test (B={B_test}µT)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../runs/angular_resolution_comparison.png", dpi=200)
    print("Saved: ../runs/angular_resolution_comparison.png")
    
    return results

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import os
    os.chdir('/home/runner/workspace/src')
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "theta45":
            run_theta45_heatmap(grid_size=21, J=0.5)
            run_theta45_heatmap(grid_size=21, J=10.0)
        elif cmd == "ridge":
            run_ridge_extraction(grid_size=31)
        elif cmd == "symmetry":
            run_symmetry_test()
        elif cmd == "angular":
            run_angular_resolution()
        elif cmd == "all":
            print("\n" + "="*60)
            print("RUNNING ALL 4 HIGH-IMPACT TESTS")
            print("="*60 + "\n")
            
            print("\n[1/4] θ=π/4 Heatmaps...")
            run_theta45_heatmap(grid_size=21, J=0.5)
            run_theta45_heatmap(grid_size=21, J=10.0)
            
            print("\n[2/4] Ridge Extraction...")
            run_ridge_extraction(grid_size=21)
            
            print("\n[3/4] Symmetry Test...")
            run_symmetry_test(n_points=15)
            
            print("\n[4/4] Angular Resolution...")
            run_angular_resolution()
            
            print("\n" + "="*60)
            print("ALL TESTS COMPLETE!")
            print("="*60)
    else:
        print("Usage: python high_impact_tests.py [theta45|ridge|symmetry|angular|all]")
