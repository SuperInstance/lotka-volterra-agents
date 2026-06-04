# lotka-volterra-agents

**Generalized Lotka-Volterra dynamics for multi-agent strategy ecology.**

[![crates.io](https://img.shields.io/crates/v/lotka-volterra-agents.svg)](https://crates.io/crates/lotka-volterra-agents)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

This crate models competitive dynamics between N agent populations using the **Lotka-Volterra competition equations** — a foundational framework from theoretical ecology that maps beautifully onto multi-agent systems.

In agent ecology, each "species" is a **strategy** (algorithm, policy, or behavior), and the populations represent the **prevalence** of each strategy in a shared environment. Competition coefficients capture how strategies interact: some coexist, some dominate, some go extinct.

## Lotka-Volterra Theory

The generalized Lotka-Volterra competition model for N species:

```
dN_i/dt = r_i · N_i · (1 - Σ_j α_ij · N_j / K_i)
```

Where:
- **N_i** — population (prevalence) of species i
- **r_i** — intrinsic growth rate of species i
- **K_i** — carrying capacity of species i
- **α_ij** — competitive effect of species j on species i

### Key Results

**Coexistence equilibrium:** At equilibrium, `A · N* = K`, where A is the interaction matrix. Solving this linear system gives the equilibrium populations.

**For 2 symmetric species with equal K:**
```
N* = K / (1 + α)
```

**Stability conditions (2-species):**
- Coexistence: `α_12 < K_1/K_2` and `α_21 < K_2/K_1`
- Species 1 wins: `α_21 > K_2/K_1` and `α_12 < K_1/K_2`
- Species 2 wins: `α_12 > K_1/K_2` and `α_21 < K_2/K_1`
- Bistability: `α_12 > K_1/K_2` and `α_21 > K_2/K_1`

### Connection to Agent Ecology

| Ecology Concept | Agent Systems Analog |
|---|---|
| Species | Strategy / algorithm |
| Population | Number of agents using strategy |
| Carrying capacity | Resource / market saturation |
| Competition coefficient | Strategy interference |
| Growth rate | Strategy adoption rate |
| Equilibrium | Stable strategy mix |
| Extinction | Strategy abandonment |

In competitive markets, reinforcement learning strategy pools, or multi-agent environments, strategies compete for "market share" just as species compete for resources. The LV framework predicts which strategies will coexist, which dominate, and how resilient the ecosystem is to perturbation.

## Features

- **`LVSystem`** — N-species competitive LV system with configurable parameters
- **`InteractionMatrix`** — Build competition matrices, check symmetry, compute eigenvalues
- **`EquilibriumFinder`** — Compute coexistence equilibria via Gaussian elimination
- **`StabilityAnalyzer`** — Jacobian analysis, eigenvalue stability classification
- **`PerturbationTest`** — Displace populations and measure recovery time / resilience
- **`Timeseries`** — Simulate N generations with RK4 integration
- **`PhasePortrait`** — Compute multiple trajectories in state space

## Quick Start

```rust
use lotka_volterra_agents::*;

// Create a 2-species competitive system
let system = LVSystem::two_species(
    1.0, 1.0,   // growth rates
    100.0, 100.0, // carrying capacities
    0.5, 0.5,   // competition coefficients (α_12, α_21)
);

// Find the coexistence equilibrium
let eq = EquilibriumFinder::find_coexistence(&system);
assert!(eq.feasible);
// N* = K / (1 + α) = 100 / 1.5 ≈ 66.67
assert!((eq.populations[0] - 100.0 / 1.5).abs() < 1e-6);

// Analyze stability
let stability = StabilityAnalyzer::analyze(&system);
assert!(stability.is_stable);

// Simulate dynamics
let ts = Timeseries::new(0.01);
let final_pops = ts.final_populations(&system, &[10.0, 20.0], 100.0);
assert!((final_pops[0] - 100.0 / 1.5).abs() < 1.0);

// Test resilience
let perturbation = PerturbationTest::new(0.1);
let result = perturbation.run(&system);
assert!(result.recovered_successfully);

// Phase portrait
let portrait = PhasePortrait::from_grid(&system, 5, 100.0, 50.0, 0.01);
assert!(portrait.all_converge(1.0));
```

## Design Principles

- **Pure Rust**, no unsafe code, minimal dependencies
- **Numerically robust**: Gaussian elimination with partial pivoting, RK4 integration
- **Zero-dep**: No external crates needed for core functionality
- **Well-tested**: 45 tests covering edge cases, known analytical results, and numerical convergence

## Installation

```toml
[dependencies]
lotka-volterra-agents = "0.1"
```

## License

MIT
