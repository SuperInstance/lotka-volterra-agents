//! Phase portrait computation for LV systems.

use crate::lv_system::LVSystem;
use crate::timeseries::Timeseries;

/// A phase portrait: multiple trajectories in state space.
#[derive(Clone, Debug)]
pub struct PhasePortrait {
    /// The trajectories (each is a list of state-space points).
    pub trajectories: Vec<Vec<Vec<f64>>>,
    /// The corresponding times for each trajectory.
    pub times: Vec<Vec<f64>>,
}

impl PhasePortrait {
    /// Compute a phase portrait by simulating from multiple initial conditions.
    ///
    /// For 2D systems, generates a grid of initial conditions. For higher
    /// dimensions, generates random initial conditions.
    pub fn from_grid(
        system: &LVSystem,
        grid_points_per_axis: usize,
        max_pop: f64,
        generations: f64,
        dt: f64,
    ) -> Self {
        let n = system.n();
        let ts = Timeseries::new(dt);
        let mut trajectories = Vec::new();
        let mut times = Vec::new();

        if n == 2 {
            // Grid in 2D
            for i in 0..grid_points_per_axis {
                for j in 0..grid_points_per_axis {
                    let n1 = max_pop * (i as f64 + 0.5) / grid_points_per_axis as f64;
                    let n2 = max_pop * (j as f64 + 0.5) / grid_points_per_axis as f64;
                    let initial = vec![n1, n2];
                    let points = ts.simulate(system, &initial, generations);
                    trajectories.push(points.iter().map(|p| p.populations.clone()).collect());
                    times.push(points.iter().map(|p| p.t).collect());
                }
            }
        } else {
            // For N != 2, generate grid_points_per_axis^2 random-ish initial conditions
            let count = grid_points_per_axis * grid_points_per_axis;
            for idx in 0..count {
                let initial: Vec<f64> = (0..n)
                    .map(|dim| {
                        let seed = ((idx * 7 + dim * 13 + 1) as f64)
                            / ((count * n) as f64);
                        max_pop * seed
                    })
                    .collect();
                let points = ts.simulate(system, &initial, generations);
                trajectories.push(points.iter().map(|p| p.populations.clone()).collect());
                times.push(points.iter().map(|p| p.t).collect());
            }
        }

        PhasePortrait {
            trajectories,
            times,
        }
    }

    /// Compute trajectories from specific initial conditions.
    pub fn from_initial_conditions(
        system: &LVSystem,
        initial_conditions: &[Vec<f64>],
        generations: f64,
        dt: f64,
    ) -> Self {
        let ts = Timeseries::new(dt);
        let mut trajectories = Vec::new();
        let mut times = Vec::new();

        for initial in initial_conditions {
            let points = ts.simulate(system, initial, generations);
            trajectories.push(points.iter().map(|p| p.populations.clone()).collect());
            times.push(points.iter().map(|p| p.t).collect());
        }

        PhasePortrait {
            trajectories,
            times,
        }
    }

    /// Get the final points of all trajectories.
    pub fn endpoints(&self) -> Vec<Vec<f64>> {
        self.trajectories
            .iter()
            .map(|traj| traj.last().cloned().unwrap_or_default())
            .collect()
    }

    /// Get the number of trajectories.
    pub fn num_trajectories(&self) -> usize {
        self.trajectories.len()
    }

    /// Check if all trajectories converge to the same point (within tolerance).
    pub fn all_converge(&self, tolerance: f64) -> bool {
        if self.trajectories.len() < 2 {
            return true;
        }
        let default = vec![];
        let ref_point = self.trajectories[0].last().unwrap_or(&default);
        for traj in &self.trajectories[1..] {
            let endpoint = traj.last().unwrap_or(&default);
            if ref_point.len() != endpoint.len() {
                return false;
            }
            for (a, b) in ref_point.iter().zip(endpoint.iter()) {
                let scale = a.abs().max(b.abs()).max(1.0);
                if (a - b).abs() / scale > tolerance {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction_matrix::InteractionMatrix;

    #[test]
    fn test_phase_portrait_grid() {
        let sys = LVSystem::two_species(1.0, 1.0, 100.0, 100.0, 0.5, 0.5);
        let portrait = PhasePortrait::from_grid(&sys, 3, 100.0, 20.0, 0.1);
        // 3x3 grid = 9 trajectories
        assert_eq!(portrait.num_trajectories(), 9);
    }

    #[test]
    fn test_phase_portrait_convergence() {
        let sys = LVSystem::two_species(1.0, 1.0, 100.0, 100.0, 0.5, 0.5);
        let portrait = PhasePortrait::from_grid(&sys, 3, 100.0, 100.0, 0.01);
        // All should converge to the same equilibrium
        assert!(portrait.all_converge(1.0));
    }

    #[test]
    fn test_phase_portrait_from_initial_conditions() {
        let sys = LVSystem::two_species(1.0, 1.0, 100.0, 100.0, 0.5, 0.5);
        let ics = vec![
            vec![10.0, 10.0],
            vec![50.0, 50.0],
            vec![90.0, 90.0],
        ];
        let portrait = PhasePortrait::from_initial_conditions(&sys, &ics, 50.0, 0.01);
        assert_eq!(portrait.num_trajectories(), 3);
    }

    #[test]
    fn test_endpoints() {
        let sys = LVSystem::two_species(1.0, 1.0, 100.0, 100.0, 0.5, 0.5);
        let ics = vec![vec![10.0, 20.0]];
        let portrait = PhasePortrait::from_initial_conditions(&sys, &ics, 100.0, 0.01);
        let endpoints = portrait.endpoints();
        assert_eq!(endpoints.len(), 1);
        let expected = 100.0 / 1.5;
        assert!((endpoints[0][0] - expected).abs() < 1.0);
    }

    #[test]
    fn test_3d_phase_portrait() {
        let interactions = InteractionMatrix::from_2d(&[
            vec![1.0, 0.2, 0.1],
            vec![0.1, 1.0, 0.2],
            vec![0.2, 0.1, 1.0],
        ]);
        let sys = LVSystem::new(
            vec![1.0, 1.0, 1.0],
            vec![100.0, 100.0, 100.0],
            interactions,
        );
        let portrait = PhasePortrait::from_grid(&sys, 2, 100.0, 50.0, 0.1);
        assert!(portrait.num_trajectories() > 0);
    }
}
