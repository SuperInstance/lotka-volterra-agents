//! Stability analysis for Lotka-Volterra equilibria.

use crate::interaction_matrix::{ComplexEigenvalue, InteractionMatrix};
use crate::lv_system::LVSystem;
use crate::equilibrium::EquilibriumFinder;

/// Report on the stability of an equilibrium point.
#[derive(Clone, Debug)]
pub struct StabilityReport {
    /// The Jacobian matrix at equilibrium.
    pub jacobian: InteractionMatrix,
    /// Eigenvalues of the Jacobian.
    pub eigenvalues: Vec<ComplexEigenvalue>,
    /// Whether the equilibrium is locally asymptotically stable.
    pub is_stable: bool,
    /// The dominant eigenvalue (largest real part).
    pub dominant_eigenvalue: ComplexEigenvalue,
    /// Stability margin: negative of the dominant real part (positive = stable).
    pub stability_margin: f64,
}

/// Analyzes the local stability of LV equilibria via Jacobian eigenvalue analysis.
pub struct StabilityAnalyzer;

impl StabilityAnalyzer {
    /// Compute the Jacobian at a given point.
    ///
    /// For the LV system dN_i/dt = r_i * N_i * (1 - sum_j alpha_ij * N_j / K_i):
    ///
    /// J_ii = r_i * (1 - sum_j alpha_ij * N_j / K_i) - r_i * N_i * alpha_ii / K_i
    /// J_ij = -r_i * N_i * alpha_ij / K_i   (i != j)
    pub fn jacobian(system: &LVSystem, populations: &[f64]) -> InteractionMatrix {
        let n = system.n();
        let mut jac = vec![0.0; n * n];

        for i in 0..n {
            let k_i = system.carrying_capacities[i];
            let r_i = system.growth_rates[i];
            let n_i = populations[i];

            let competition: f64 = (0..n)
                .map(|j| system.interactions.get(i, j) * populations[j])
                .sum();

            // Diagonal: r_i * (1 - C/K) - r_i * N_i * alpha_ii / K_i
            jac[i * n + i] = r_i * (1.0 - competition / k_i)
                - r_i * n_i * system.interactions.get(i, i) / k_i;

            // Off-diagonal: -r_i * N_i * alpha_ij / K_i
            for j in 0..n {
                if j != i {
                    jac[i * n + j] = -r_i * n_i * system.interactions.get(i, j) / k_i;
                }
            }
        }

        InteractionMatrix::from_slice(n, &jac)
    }

    /// Perform full stability analysis at the coexistence equilibrium.
    pub fn analyze(system: &LVSystem) -> StabilityReport {
        let eq_result = EquilibriumFinder::find_coexistence(system);

        if !eq_result.feasible {
            return StabilityReport {
                jacobian: InteractionMatrix::identity(0),
                eigenvalues: vec![],
                is_stable: false,
                dominant_eigenvalue: ComplexEigenvalue {
                    real: f64::NAN,
                    imag: 0.0,
                },
                stability_margin: f64::NAN,
            };
        }

        Self::analyze_at(system, &eq_result.populations)
    }

    /// Analyze stability at a specific point.
    pub fn analyze_at(system: &LVSystem, populations: &[f64]) -> StabilityReport {
        let jac = Self::jacobian(system, populations);
        let eigenvalues = jac.eigenvalues();

        let dominant = eigenvalues
            .iter()
            .max_by(|a, b| a.real.partial_cmp(&b.real).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
            .unwrap_or(ComplexEigenvalue {
                real: f64::NAN,
                imag: 0.0,
            });

        let is_stable = eigenvalues.iter().all(|e| e.real < 0.0);
        let stability_margin = -dominant.real;

        StabilityReport {
            jacobian: jac,
            eigenvalues,
            is_stable,
            dominant_eigenvalue: dominant,
            stability_margin,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction_matrix::InteractionMatrix;

    #[test]
    fn test_single_species_stable() {
        let sys = LVSystem::new(
            vec![1.0],
            vec![100.0],
            InteractionMatrix::identity(1),
        );
        let report = StabilityAnalyzer::analyze(&sys);
        assert!(report.is_stable);
        assert!(report.stability_margin > 0.0);
    }

    #[test]
    fn test_two_species_stable_coexistence() {
        // Weak competition: both alphas < 1 => stable coexistence
        let sys = LVSystem::two_species(1.0, 1.0, 100.0, 100.0, 0.5, 0.5);
        let report = StabilityAnalyzer::analyze(&sys);
        assert!(report.is_stable);
        assert!(report.stability_margin > 0.0);
    }

    #[test]
    fn test_jacobian_at_equilibrium() {
        let sys = LVSystem::new(
            vec![1.0],
            vec![100.0],
            InteractionMatrix::identity(1),
        );
        let jac = StabilityAnalyzer::jacobian(&sys, &[100.0]);
        // At K, J = r * (1 - 1) - r * K * 1 / K = -r = -1
        assert!((jac.get(0, 0) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_jacobian_off_diagonal() {
        let sys = LVSystem::two_species(1.0, 1.0, 100.0, 100.0, 0.5, 0.5);
        let eq = vec![100.0 / 1.5, 100.0 / 1.5];
        let jac = StabilityAnalyzer::jacobian(&sys, &eq);
        // Off-diagonal should be negative
        assert!(jac.get(0, 1) < 0.0);
        assert!(jac.get(1, 0) < 0.0);
    }

    #[test]
    fn test_stability_margin() {
        let sys = LVSystem::new(
            vec![2.0],
            vec![100.0],
            InteractionMatrix::identity(1),
        );
        let report = StabilityAnalyzer::analyze(&sys);
        // J at K = -r = -2, so margin = 2
        assert!((report.stability_margin - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_infeasible_returns_unstable() {
        // Strong mutual exclusion
        let sys = LVSystem::two_species(1.0, 1.0, 100.0, 100.0, 3.0, 3.0);
        let report = StabilityAnalyzer::analyze(&sys);
        assert!(!report.is_stable);
    }
}
