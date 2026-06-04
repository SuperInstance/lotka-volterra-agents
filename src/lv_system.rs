//! The core Lotka-Volterra system definition.

use crate::interaction_matrix::InteractionMatrix;

/// A Lotka-Volterra competition system with N species.
///
/// The dynamics follow:
/// ```text
/// dN_i/dt = r_i * N_i * (1 - (sum_j alpha_ij * N_j) / K_i)
/// ```
///
/// where `r_i` is the intrinsic growth rate, `K_i` is the carrying capacity,
/// and `alpha_ij` is the competitive effect of species j on species i.
#[derive(Clone, Debug)]
pub struct LVSystem {
    /// Intrinsic growth rates for each species.
    pub growth_rates: Vec<f64>,
    /// Carrying capacities for each species.
    pub carrying_capacities: Vec<f64>,
    /// Competition/interaction matrix.
    pub interactions: InteractionMatrix,
}

impl LVSystem {
    /// Create a new LV system.
    ///
    /// Panics if dimensions don't match.
    pub fn new(
        growth_rates: Vec<f64>,
        carrying_capacities: Vec<f64>,
        interactions: InteractionMatrix,
    ) -> Self {
        let n = growth_rates.len();
        assert_eq!(carrying_capacities.len(), n, "Dimension mismatch: carrying_capacities");
        assert_eq!(interactions.n(), n, "Dimension mismatch: interactions matrix");
        Self {
            growth_rates,
            carrying_capacities,
            interactions,
        }
    }

    /// Create a 2-species system with standard competition coefficients.
    pub fn two_species(
        r1: f64, r2: f64,
        k1: f64, k2: f64,
        alpha_12: f64, alpha_21: f64,
    ) -> Self {
        Self {
            growth_rates: vec![r1, r2],
            carrying_capacities: vec![k1, k2],
            interactions: InteractionMatrix::two_species(alpha_12, alpha_21),
        }
    }

    /// Number of species.
    pub fn n(&self) -> usize {
        self.growth_rates.len()
    }

    /// Compute the growth rates dN/dt at given populations.
    pub fn derivatives(&self, populations: &[f64]) -> Vec<f64> {
        assert_eq!(populations.len(), self.n(), "Population vector dimension mismatch");
        (0..self.n())
            .map(|i| {
                let competition: f64 = (0..self.n())
                    .map(|j| self.interactions.get(i, j) * populations[j])
                    .sum();
                let k_i = self.carrying_capacities[i];
                let effective_k = if k_i.abs() < 1e-15 { 1e-15 } else { k_i };
                self.growth_rates[i] * populations[i] * (1.0 - competition / effective_k)
            })
            .collect()
    }

    /// Step the system forward by dt using RK4 integration.
    pub fn step_rk4(&self, populations: &[f64], dt: f64) -> Vec<f64> {
        let n = self.n();

        let k1 = self.derivatives(populations);

        let p2: Vec<f64> = (0..n).map(|i| populations[i] + 0.5 * dt * k1[i]).collect();
        let k2 = self.derivatives(&p2);

        let p3: Vec<f64> = (0..n).map(|i| populations[i] + 0.5 * dt * k2[i]).collect();
        let k3 = self.derivatives(&p3);

        let p4: Vec<f64> = (0..n).map(|i| populations[i] + dt * k3[i]).collect();
        let k4 = self.derivatives(&p4);

        (0..n)
            .map(|i| {
                let dp = (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
                (populations[i] + dt * dp).max(0.0) // populations can't go negative
            })
            .collect()
    }

    /// Step forward using Euler method (less accurate, faster).
    pub fn step_euler(&self, populations: &[f64], dt: f64) -> Vec<f64> {
        let derivs = self.derivatives(populations);
        (0..self.n())
            .map(|i| (populations[i] + dt * derivs[i]).max(0.0))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derivatives_zero_population() {
        let sys = LVSystem::two_species(1.0, 1.0, 100.0, 100.0, 0.5, 0.5);
        let derivs = sys.derivatives(&[0.0, 0.0]);
        assert!((derivs[0]).abs() < 1e-10);
        assert!((derivs[1]).abs() < 1e-10);
    }

    #[test]
    fn test_derivatives_at_carrying_capacity() {
        // Single species at K should have zero growth
        let sys = LVSystem::new(
            vec![1.0],
            vec![100.0],
            InteractionMatrix::identity(1),
        );
        let derivs = sys.derivatives(&[100.0]);
        assert!(derivs[0].abs() < 1e-10);
    }

    #[test]
    fn test_derivatives_below_carrying_capacity() {
        let sys = LVSystem::new(
            vec![1.0],
            vec![100.0],
            InteractionMatrix::identity(1),
        );
        let derivs = sys.derivatives(&[50.0]);
        assert!(derivs[0] > 0.0); // should be growing
    }

    #[test]
    fn test_two_species_derivatives_symmetric() {
        let sys = LVSystem::two_species(1.0, 1.0, 100.0, 100.0, 0.5, 0.5);
        let derivs = sys.derivatives(&[10.0, 10.0]);
        // With symmetric params and equal pops, derivatives should be equal
        assert!((derivs[0] - derivs[1]).abs() < 1e-10);
    }

    #[test]
    fn test_step_rk4_single_species() {
        let sys = LVSystem::new(
            vec![1.0],
            vec![100.0],
            InteractionMatrix::identity(1),
        );
        let p = vec![10.0];
        let p_new = sys.step_rk4(&p, 0.1);
        assert!(p_new[0] > 10.0); // should grow
        assert!(p_new[0] < 100.0); // shouldn't exceed K in one step
    }

    #[test]
    fn test_step_euler_single_species() {
        let sys = LVSystem::new(
            vec![1.0],
            vec![100.0],
            InteractionMatrix::identity(1),
        );
        let p = vec![10.0];
        let p_new = sys.step_euler(&p, 0.1);
        assert!(p_new[0] > 10.0);
    }

    #[test]
    fn test_no_negative_populations() {
        let sys = LVSystem::two_species(1.0, 1.0, 100.0, 100.0, 5.0, 5.0);
        // Strong competition with very different pops
        let p_new = sys.step_rk4(&[1.0, 200.0], 10.0);
        assert!(p_new[0] >= 0.0);
        assert!(p_new[1] >= 0.0);
    }
}
