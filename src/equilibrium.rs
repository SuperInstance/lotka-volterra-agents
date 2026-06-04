//! Equilibrium finding for Lotka-Volterra systems.

use crate::lv_system::LVSystem;

/// Result of equilibrium computation.
#[derive(Clone, Debug)]
pub struct EquilibriumResult {
    /// The equilibrium populations (N*).
    pub populations: Vec<f64>,
    /// Whether all populations are strictly positive (feasible coexistence).
    pub feasible: bool,
    /// Whether the equilibrium is internally consistent (residuals near zero).
    pub verified: bool,
}

/// Finds and analyzes equilibrium points of LV systems.
pub struct EquilibriumFinder;

impl EquilibriumFinder {
    /// Compute the coexistence equilibrium analytically.
    ///
    /// At equilibrium, dN_i/dt = 0 for all i, which gives:
    /// ```text
    /// sum_j alpha_ij * N_j = K_i   for each i
    /// ```
    /// This is a linear system: A * N* = K
    ///
    /// Solves via Gaussian elimination with partial pivoting.
    pub fn find_coexistence(system: &LVSystem) -> EquilibriumResult {
        let n = system.n();
        let mut augmented = vec![0.0; n * (n + 1)];

        // Build augmented matrix [A | K]
        for i in 0..n {
            for j in 0..n {
                augmented[i * (n + 1) + j] = system.interactions.get(i, j);
            }
            augmented[i * (n + 1) + n] = system.carrying_capacities[i];
        }

        // Gaussian elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            let mut max_val = augmented[col * (n + 1) + col].abs();
            for row in (col + 1)..n {
                let val = augmented[row * (n + 1) + col].abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }

            if max_val < 1e-14 {
                // Singular matrix - no unique equilibrium
                return EquilibriumResult {
                    populations: vec![f64::NAN; n],
                    feasible: false,
                    verified: false,
                };
            }

            // Swap rows
            if max_row != col {
                for j in 0..=n {
                    let (src, dst) = (max_row * (n + 1) + j, col * (n + 1) + j);
                    let tmp = augmented[src];
                    augmented[src] = augmented[dst];
                    augmented[dst] = tmp;
                }
            }

            // Eliminate below
            for row in (col + 1)..n {
                let factor = augmented[row * (n + 1) + col] / augmented[col * (n + 1) + col];
                for j in col..=n {
                    augmented[row * (n + 1) + j] -= factor * augmented[col * (n + 1) + j];
                }
            }
        }

        // Back substitution
        let mut populations = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = augmented[i * (n + 1) + n];
            for j in (i + 1)..n {
                sum -= augmented[i * (n + 1) + j] * populations[j];
            }
            populations[i] = sum / augmented[i * (n + 1) + i];
        }

        // Check feasibility: all populations > 0
        let feasible = populations.iter().all(|&p| p > 0.0 && p.is_finite());

        // Verify: check that A * N* ≈ K
        let verified = Self::verify_equilibrium(system, &populations);

        EquilibriumResult {
            populations,
            feasible,
            verified,
        }
    }

    /// Verify that the equilibrium satisfies A*N = K within tolerance.
    pub fn verify_equilibrium(system: &LVSystem, equilibrium: &[f64]) -> bool {
        let n = system.n();
        let tol = 1e-6;
        for i in 0..n {
            let sum: f64 = (0..n)
                .map(|j| system.interactions.get(i, j) * equilibrium[j])
                .sum();
            if (sum - system.carrying_capacities[i]).abs() > tol {
                return false;
            }
        }
        true
    }

    /// Compute the 2-species equilibrium using the known formula.
    ///
    /// For the standard 2-species competition model with carrying capacities K1, K2
    /// and competition coefficients alpha_12, alpha_21:
    ///
    /// N1* = K1 * (1 - alpha_12 * K2 / K1) / (1 - alpha_12 * alpha_21)
    /// N2* = K2 * (1 - alpha_21 * K1 / K2) / (1 - alpha_12 * alpha_21)
    ///
    /// Wait, more precisely for the matrix form A * N = K:
    /// [[1, alpha_12], [alpha_21, 1]] * [N1, N2] = [K1, K2]
    ///
    /// N1* = (K1 - alpha_12 * K2) / (1 - alpha_12 * alpha_21)
    /// N2* = (K2 - alpha_21 * K1) / (1 - alpha_12 * alpha_21)
    pub fn two_species_equilibrium(
        k1: f64, k2: f64,
        alpha_12: f64, alpha_21: f64,
    ) -> (f64, f64) {
        let det = 1.0 - alpha_12 * alpha_21;
        if det.abs() < 1e-14 {
            return (f64::NAN, f64::NAN);
        }
        let n1 = (k1 - alpha_12 * k2) / det;
        let n2 = (k2 - alpha_21 * k1) / det;
        (n1, n2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction_matrix::InteractionMatrix;

    #[test]
    fn test_single_species_equilibrium() {
        let sys = LVSystem::new(
            vec![1.0],
            vec![100.0],
            InteractionMatrix::identity(1),
        );
        let result = EquilibriumFinder::find_coexistence(&sys);
        assert!((result.populations[0] - 100.0).abs() < 1e-6);
        assert!(result.feasible);
        assert!(result.verified);
    }

    #[test]
    fn test_two_species_equilibrium_symmetric() {
        let sys = LVSystem::two_species(1.0, 1.0, 100.0, 100.0, 0.5, 0.5);
        let result = EquilibriumFinder::find_coexistence(&sys);
        // With symmetric competition, N1* = N2* = K/(1+alpha) = 100/1.5 ≈ 66.67
        let expected = 100.0 / 1.5;
        assert!((result.populations[0] - expected).abs() < 1e-6);
        assert!((result.populations[1] - expected).abs() < 1e-6);
        assert!(result.feasible);
        assert!(result.verified);
    }

    #[test]
    fn test_two_species_formula() {
        let (n1, n2) = EquilibriumFinder::two_species_equilibrium(100.0, 100.0, 0.5, 0.5);
        let expected = 100.0 / 1.5;
        assert!((n1 - expected).abs() < 1e-10);
        assert!((n2 - expected).abs() < 1e-10);
    }

    #[test]
    fn test_two_species_equilibrium_asymmetric() {
        let sys = LVSystem::two_species(1.0, 1.0, 100.0, 80.0, 0.3, 0.6);
        let result = EquilibriumFinder::find_coexistence(&sys);
        // det = 1 - 0.3*0.6 = 0.82
        // N1* = (100 - 0.3*80) / 0.82 = 76/0.82 ≈ 92.68
        // N2* = (80 - 0.6*100) / 0.82 = 20/0.82 ≈ 24.39
        assert!((result.populations[0] - 76.0 / 0.82).abs() < 1e-6);
        assert!((result.populations[1] - 20.0 / 0.82).abs() < 1e-6);
        assert!(result.feasible);
    }

    #[test]
    fn test_infeasible_equilibrium() {
        // Strong asymmetric competition: species 1 excluded
        // alpha_12 = 1.5, alpha_21 = 0.8, K1 = 50, K2 = 100
        // N1* = (50 - 1.5*100) / (1 - 1.5*0.8) = (50-150)/(1-1.2) = -100/-0.2 = 500
        // N2* = (100 - 0.8*50) / (1 - 1.5*0.8) = (100-40)/-0.2 = -300
        let sys = LVSystem::two_species(1.0, 1.0, 50.0, 100.0, 1.5, 0.8);
        let result = EquilibriumFinder::find_coexistence(&sys);
        // N2* should be negative => infeasible
        assert!(!result.feasible);
    }

    #[test]
    fn test_three_species_equilibrium() {
        let interactions = InteractionMatrix::from_2d(&[
            vec![1.0, 0.2, 0.1],
            vec![0.2, 1.0, 0.3],
            vec![0.1, 0.3, 1.0],
        ]);
        let sys = LVSystem::new(
            vec![1.0, 1.0, 1.0],
            vec![100.0, 100.0, 100.0],
            interactions,
        );
        let result = EquilibriumFinder::find_coexistence(&sys);
        assert!(result.feasible);
        assert!(result.verified);
        // All should be less than K=100 due to competition
        for p in &result.populations {
            assert!(p < &100.0);
        }
    }

    #[test]
    fn test_known_two_species_k_over_1_plus_alpha() {
        // Classic result: for symmetric 2-species with equal K,
        // N* = K / (1 + alpha)
        let k = 150.0;
        let alpha = 0.4;
        let sys = LVSystem::two_species(0.5, 0.5, k, k, alpha, alpha);
        let result = EquilibriumFinder::find_coexistence(&sys);
        let expected = k / (1.0 + alpha);
        assert!((result.populations[0] - expected).abs() < 1e-6);
        assert!((result.populations[1] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_verify_equilibrium() {
        let sys = LVSystem::two_species(1.0, 1.0, 100.0, 100.0, 0.5, 0.5);
        let eq = vec![100.0 / 1.5, 100.0 / 1.5];
        assert!(EquilibriumFinder::verify_equilibrium(&sys, &eq));

        let bad = vec![50.0, 50.0];
        assert!(!EquilibriumFinder::verify_equilibrium(&sys, &bad));
    }
}
