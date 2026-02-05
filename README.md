# Radical Pair Mechanism: Universal Shelves in Quantum Magnetic Sensing

Simulation package for investigating "shelves" (structured response surfaces) in the Radical Pair Mechanism (RPM) for quantum magnetoreception.

## Key Findings

1. **Universal Collapse**: All response curves collapse onto a single universal form when centered and normalized (correlation > 0.99)
2. **Ratio Locking**: Peak position locked at kS/kT ≈ 8-10, invariant to hyperfine coupling A and exchange J
3. **Amplitude Scaling**: Peak amplitude scales as A^1.4 (R² = 0.99)
4. **Single Universal Shelf**: One shelf found across 4 decades in kS/kT
5. **Hyperfine Control**: Shelf structure bifurcates at finite hyperfine mixing (disappears at A=0)
6. **Angular Drift**: Field-dependent angular oscillation at high kS/kT (Zeeman-hyperfine interference)

## Files

### Core Simulation
- `rpm_shelves.py` - Main simulation engine with Zeeman, hyperfine, and exchange coupling Hamiltonians
- `high_impact_tests.py` - High-impact analysis scripts

### Figures (paper_figures/)

| Figure | Description |
|--------|-------------|
| `dual_heatmap_figure1.png` | 20× contrast between optimal (J=0.5) and suppressed (J=10) regimes |
| `ablation_test.png` | Exchange coupling J as control knob |
| `robustness_shelf.png` | Noise-assisted response window |
| `scaling_shelf_analysis.png` | Scale invariance analysis |
| `multilayer_scaling.png` | Multi-layer A comparison |
| `mirror_symmetry.png` | 95% symmetric power in log-ratio space |
| `universal_collapse.png` | Universal curve collapse (corr 0.99+) |
| `universal_locking_final.png` | A^1.4 amplitude scaling with locked peak |
| `extended_range.png` | Single shelf across 4 decades |
| `proton_test.png` | Proton ON/OFF comparison |
| `ablation_A_curve.png` | Continuous hyperfine ablation |
| `ghost_shell_analysis.png` | Face-to-face ghost shell analysis |
| `angular_drift.png` | Angular drift in (B,θ) space |
| `SI_recombination_models.png` | Haberkorn vs Lindblad model comparison |

## Physical Model

The simulation solves the Liouville-von Neumann equation for a radical pair with:
- Zeeman interaction with external magnetic field B
- Isotropic hyperfine coupling A with a single nuclear spin
- Exchange coupling J between electron spins
- Spin-selective recombination (singlet rate kS, triplet rate kT)
- Optional Lindblad decoherence

## Key Parameter Regimes

- **Optimal**: J = 0.5, produces maximum anisotropy
- **Suppressed**: J = 10, kills signal by 20×
- **Peak ratio**: kS/kT ≈ 8-10 (universal, locked)

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib

## Citation

[Paper details to be added upon publication]
