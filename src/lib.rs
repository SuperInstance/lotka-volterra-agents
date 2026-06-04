//! # lotka-volterra-agents
//!
//! Generalized Lotka-Volterra dynamics for multi-agent strategy ecology.
//!
//! This crate provides tools for modeling competitive dynamics between N agent
//! populations using the Lotka-Volterra competition equations. It supports
//! interaction matrix construction, equilibrium finding, stability analysis,
//! perturbation testing, timeseries simulation, and phase portrait computation.

mod interaction_matrix;
mod lv_system;
mod equilibrium;
mod stability;
mod perturbation;
mod timeseries;
mod phase_portrait;

pub use interaction_matrix::InteractionMatrix;
pub use lv_system::LVSystem;
pub use equilibrium::{EquilibriumFinder, EquilibriumResult};
pub use stability::{StabilityAnalyzer, StabilityReport};
pub use perturbation::{PerturbationTest, PerturbationResult};
pub use timeseries::{Timeseries, TimeseriesPoint};
pub use phase_portrait::PhasePortrait;
