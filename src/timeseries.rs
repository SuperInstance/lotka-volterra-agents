//! Timeseries simulation of LV dynamics.

use crate::lv_system::LVSystem;

/// A single point in a timeseries.
#[derive(Clone, Debug)]
pub struct TimeseriesPoint {
    /// Time (in generations).
    pub t: f64,
    /// Population values at this time.
    pub populations: Vec<f64>,
}

/// Simulates LV dynamics over time.
pub struct Timeseries {
    /// Integration timestep.
    pub dt: f64,
}

impl Default for Timeseries {
    fn default() -> Self {
        Self { dt: 0.01 }
    }
}

impl Timeseries {
    pub fn new(dt: f64) -> Self {
        Self { dt }
    }

    /// Simulate for N generations, returning all sampled points.
    pub fn simulate(&self, system: &LVSystem, initial: &[f64], generations: f64) -> Vec<TimeseriesPoint> {
        let mut points = Vec::new();
        let mut current = initial.to_vec();
        let steps = (generations / self.dt) as usize;

        points.push(TimeseriesPoint {
            t: 0.0,
            populations: current.clone(),
        });

        for step in 1..=steps {
            current = system.step_rk4(&current, self.dt);
            points.push(TimeseriesPoint {
                t: step as f64 * self.dt,
                populations: current.clone(),
            });
        }

        points
    }

    /// Simulate and sample at regular intervals (save memory for long runs).
    pub fn simulate_sampled(
        &self,
        system: &LVSystem,
        initial: &[f64],
        generations: f64,
        sample_interval: f64,
    ) -> Vec<TimeseriesPoint> {
        let mut points = Vec::new();
        let mut current = initial.to_vec();
        let steps = (generations / self.dt) as usize;
        let sample_every = (sample_interval / self.dt).max(1.0) as usize;

        points.push(TimeseriesPoint {
            t: 0.0,
            populations: current.clone(),
        });

        for step in 1..=steps {
            current = system.step_rk4(&current, self.dt);
            if step % sample_every == 0 || step == steps {
                points.push(TimeseriesPoint {
                    t: step as f64 * self.dt,
                    populations: current.clone(),
                });
            }
        }

        points
    }

    /// Get the final populations after simulation.
    pub fn final_populations(&self, system: &LVSystem, initial: &[f64], generations: f64) -> Vec<f64> {
        let steps = (generations / self.dt) as usize;
        let mut current = initial.to_vec();
        for _ in 0..steps {
            current = system.step_rk4(&current, self.dt);
        }
        current
    }

    /// Check if populations converge to equilibrium within tolerance.
    pub fn converges(
        &self,
        system: &LVSystem,
        initial: &[f64],
        generations: f64,
        equilibrium: &[f64],
        tolerance: f64,
    ) -> bool {
        let final_pops = self.final_populations(system, initial, generations);
        for i in 0..final_pops.len() {
            let scale = equilibrium[i].abs().max(1.0);
            if (final_pops[i] - equilibrium[i]).abs() / scale > tolerance {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction_matrix::InteractionMatrix;
    use crate::equilibrium::EquilibriumFinder;

    #[test]
    fn test_single_species_converges() {
        let sys = LVSystem::new(
            vec![1.0],
            vec![100.0],
            InteractionMatrix::identity(1),
        );
        let ts = Timeseries::new(0.01);
        let final_pops = ts.final_populations(&sys, &[10.0], 50.0);
        assert!((final_pops[0] - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_simulate_length() {
        let sys = LVSystem::two_species(1.0, 1.0, 100.0, 100.0, 0.5, 0.5);
        let ts = Timeseries::new(0.1);
        let points = ts.simulate(&sys, &[10.0, 10.0], 10.0);
        // 10/0.1 = 100 steps + initial = 101 points
        assert_eq!(points.len(), 101);
    }

    #[test]
    fn test_sampled_timeseries() {
        let sys = LVSystem::two_species(1.0, 1.0, 100.0, 100.0, 0.5, 0.5);
        let ts = Timeseries::new(0.01);
        let points = ts.simulate_sampled(&sys, &[10.0, 10.0], 10.0, 1.0);
        // Should have ~11 points (0, 1, 2, ..., 10) plus final
        assert!(points.len() >= 10);
        assert!(points.len() <= 12);
    }

    #[test]
    fn test_converges_to_equilibrium() {
        let sys = LVSystem::two_species(1.0, 1.0, 100.0, 100.0, 0.5, 0.5);
        let eq = EquilibriumFinder::find_coexistence(&sys);
        let ts = Timeseries::new(0.01);
        assert!(ts.converges(&sys, &[10.0, 10.0], 100.0, &eq.populations, 0.01));
    }

    #[test]
    fn test_two_species_timeseries_convergence() {
        let sys = LVSystem::two_species(1.0, 1.0, 100.0, 100.0, 0.5, 0.5);
        let ts = Timeseries::new(0.01);
        let final_pops = ts.final_populations(&sys, &[10.0, 20.0], 100.0);
        let expected = 100.0 / 1.5;
        assert!((final_pops[0] - expected).abs() < 1.0);
        assert!((final_pops[1] - expected).abs() < 1.0);
    }

    #[test]
    fn test_monotonic_approach_single_species() {
        let sys = LVSystem::new(
            vec![1.0],
            vec![100.0],
            InteractionMatrix::identity(1),
        );
        let ts = Timeseries::new(0.1);
        let points = ts.simulate(&sys, &[10.0], 20.0);
        // Starting below K, should monotonically increase (for logistic growth)
        for i in 1..points.len() {
            assert!(points[i].populations[0] >= points[i - 1].populations[0] - 0.1);
        }
    }
}
