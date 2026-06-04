//! Perturbation testing and resilience scoring.

use crate::lv_system::LVSystem;
use crate::equilibrium::EquilibriumFinder;

/// Result of a perturbation test.
#[derive(Clone, Debug)]
pub struct PerturbationResult {
    /// The perturbed starting populations.
    pub perturbed: Vec<f64>,
    /// Populations after recovery period.
    pub recovered: Vec<f64>,
    /// Equilibrium populations (target).
    pub equilibrium: Vec<f64>,
    /// Maximum absolute deviation from equilibrium during recovery.
    pub max_deviation: f64,
    /// Time to recover within tolerance (in generations).
    pub recovery_time: Option<f64>,
    /// Resilience score (0-1, higher is better). Computed as 1/(1+recovery_time).
    pub resilience_score: f64,
    /// Whether the system recovered within the simulation window.
    pub recovered_successfully: bool,
}

/// Tests the resilience of LV equilibria to perturbations.
pub struct PerturbationTest {
    /// Perturbation magnitude as a fraction of equilibrium (default 0.1 = 10%).
    pub magnitude: f64,
    /// Integration timestep.
    pub dt: f64,
    /// Maximum simulation time for recovery.
    pub max_time: f64,
    /// Tolerance for "recovered" (fraction of equilibrium).
    pub tolerance: f64,
}

impl Default for PerturbationTest {
    fn default() -> Self {
        Self {
            magnitude: 0.1,
            dt: 0.01,
            max_time: 100.0,
            tolerance: 0.01,
        }
    }
}

impl PerturbationTest {
    pub fn new(magnitude: f64) -> Self {
        Self {
            magnitude,
            ..Default::default()
        }
    }

    /// Run a perturbation test: displace populations and measure recovery.
    ///
    /// Applies a multiplicative perturbation to each species and simulates
    /// forward, tracking when (if ever) the system returns within tolerance
    /// of the equilibrium.
    pub fn run(&self, system: &LVSystem) -> PerturbationResult {
        let eq_result = EquilibriumFinder::find_coexistence(system);

        if !eq_result.feasible {
            return PerturbationResult {
                perturbed: vec![],
                recovered: vec![],
                equilibrium: vec![],
                max_deviation: f64::NAN,
                recovery_time: None,
                resilience_score: 0.0,
                recovered_successfully: false,
            };
        }

        let eq = &eq_result.populations;
        let n = system.n();

        // Apply perturbation: multiply each by (1 + magnitude) with alternating signs
        let perturbed: Vec<f64> = (0..n)
            .map(|i| {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                (eq[i] * (1.0 + sign * self.magnitude)).max(0.01)
            })
            .collect();

        // Simulate and track recovery
        let mut current = perturbed.clone();
        let mut max_deviation = 0.0_f64;
        let mut recovery_time: Option<f64> = None;
        let steps = (self.max_time / self.dt) as usize;

        for step in 0..steps {
            let t = step as f64 * self.dt;

            // Check deviation
            let max_rel_dev: f64 = (0..n)
                .map(|i| {
                    let dev = (current[i] - eq[i]).abs();
                    let scale = eq[i].abs().max(1.0);
                    dev / scale
                })
                .fold(0.0_f64, |a, b| a.max(b));

            if max_rel_dev > max_deviation {
                max_deviation = max_rel_dev;
            }

            // Check if recovered
            if max_rel_dev < self.tolerance && recovery_time.is_none() {
                recovery_time = Some(t);
                break;
            }

            current = system.step_rk4(&current, self.dt);
        }

        let recovered_successfully = recovery_time.is_some();
        let resilience_score = match recovery_time {
            Some(t) => 1.0 / (1.0 + t),
            None => 0.0,
        };

        PerturbationResult {
            perturbed,
            recovered: current,
            equilibrium: eq.clone(),
            max_deviation,
            recovery_time,
            resilience_score,
            recovered_successfully,
        }
    }

    /// Run a custom perturbation with a specific displacement vector.
    pub fn run_custom(&self, system: &LVSystem, displacement: &[f64]) -> PerturbationResult {
        let eq_result = EquilibriumFinder::find_coexistence(system);
        if !eq_result.feasible {
            return PerturbationResult {
                perturbed: vec![],
                recovered: vec![],
                equilibrium: vec![],
                max_deviation: f64::NAN,
                recovery_time: None,
                resilience_score: 0.0,
                recovered_successfully: false,
            };
        }

        let eq = &eq_result.populations;
        let perturbed: Vec<f64> = (0..system.n())
            .map(|i| (eq[i] + displacement[i]).max(0.01))
            .collect();

        let mut current = perturbed.clone();
        let mut max_deviation = 0.0_f64;
        let mut recovery_time: Option<f64> = None;
        let steps = (self.max_time / self.dt) as usize;

        for step in 0..steps {
            let t = step as f64 * self.dt;
            let max_rel_dev: f64 = (0..system.n())
                .map(|i| (current[i] - eq[i]).abs() / eq[i].abs().max(1.0))
                .fold(0.0_f64, |a, b| a.max(b));

            if max_rel_dev > max_deviation {
                max_deviation = max_rel_dev;
            }

            if max_rel_dev < self.tolerance && recovery_time.is_none() {
                recovery_time = Some(t);
                break;
            }

            current = system.step_rk4(&current, self.dt);
        }

        let recovered_successfully = recovery_time.is_some();
        let resilience_score = match recovery_time {
            Some(t) => 1.0 / (1.0 + t),
            None => 0.0,
        };

        PerturbationResult {
            perturbed,
            recovered: current,
            equilibrium: eq.clone(),
            max_deviation,
            recovery_time,
            resilience_score,
            recovered_successfully,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction_matrix::InteractionMatrix;

    #[test]
    fn test_single_species_recovery() {
        let sys = LVSystem::new(
            vec![1.0],
            vec![100.0],
            InteractionMatrix::identity(1),
        );
        let test = PerturbationTest::new(0.2);
        let result = test.run(&sys);
        assert!(result.recovered_successfully);
        assert!(result.resilience_score > 0.0);
    }

    #[test]
    fn test_two_species_recovery() {
        let sys = LVSystem::two_species(1.0, 1.0, 100.0, 100.0, 0.5, 0.5);
        let test = PerturbationTest::new(0.1);
        let result = test.run(&sys);
        assert!(result.recovered_successfully);
    }

    #[test]
    fn test_perturbed_values() {
        let sys = LVSystem::two_species(1.0, 1.0, 100.0, 100.0, 0.5, 0.5);
        let test = PerturbationTest::new(0.2);
        let result = test.run(&sys);
        let eq = 100.0 / 1.5;
        // Species 0 gets +20%, species 1 gets -20%
        assert!((result.perturbed[0] - eq * 1.2).abs() < 1e-6);
        assert!((result.perturbed[1] - eq * 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_faster_growth_faster_recovery() {
        let sys_slow = LVSystem::two_species(0.5, 0.5, 100.0, 100.0, 0.5, 0.5);
        let sys_fast = LVSystem::two_species(2.0, 2.0, 100.0, 100.0, 0.5, 0.5);
        let test = PerturbationTest::new(0.1);
        let slow = test.run(&sys_slow);
        let fast = test.run(&sys_fast);
        // Faster growth should recover sooner
        assert!(fast.recovery_time.unwrap() < slow.recovery_time.unwrap());
    }

    #[test]
    fn test_custom_perturbation() {
        let sys = LVSystem::new(
            vec![1.0],
            vec![100.0],
            InteractionMatrix::identity(1),
        );
        let test = PerturbationTest::default();
        let result = test.run_custom(&sys, &[10.0]);
        assert!(result.recovered_successfully);
        assert!((result.perturbed[0] - 110.0).abs() < 1e-6);
    }
}
