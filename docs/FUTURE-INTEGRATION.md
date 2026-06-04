# Future Integration: lotka-volterra-agents

## Current State
Generalized Lotka-Volterra dynamics for multi-agent strategy ecology. Models N competing agent populations with competition coefficients, carrying capacities, and intrinsic growth rates. Published on crates.io. Proves that coexistence requires intra-specific competition > inter-specific competition.

## Integration Opportunities

### With room competition
When multiple rooms compete for shared resources (compute, LLM proxy budget, GPU time), Lotka-Volterra models the competition. Rooms are "species" with different resource needs. The competition coefficients capture how much one room's usage affects another. Coexistence is ensured when rooms specialize (intra-specific competition > inter-specific).

### With strategy-ecology
The 5 strategy species (Explorer, Diplomat, Marksman, Climber, Prospector) from strategy-ecology are modeled by lotka-volterra-agents' N-species competition equations. The coexistence result (100% ecological resilience) comes from the LV parameters. lotka-volterra-agents provides the math; strategy-ecology provides the species definitions.

### With ternary-cell population dynamics
Lotka-Volterra dynamics govern how cell populations within a room coexist. Different cell types (strategies) compete for energy. The GC phase uses LV equations to decide which cells to keep: cells with high intra-specific competition are culled; cells with high inter-specific cooperation are retained.

## Dormant Ideas Now Unlockable
The LV models were standalone ecological simulations. Now room cell populations provide the concrete application: managing diversity in ternary grids through principled ecological dynamics rather than ad-hoc heuristics.

## Potential in Mature Systems
Lotka-Volterra is the fleet's population management theory. At every scale (cells within rooms, rooms within the fleet, strategies within cells), LV dynamics ensure coexistence, diversity, and resilience. The fleet IS an ecosystem.

## Cross-Pollination Ideas
- **strategy-ecology**: Species definitions for LV populations
- **conservation-matrix-rs**: Law 3 (species coexist) proven by LV dynamics
- **evolution-ternary**: Evolution + ecology = the complete population dynamics model
- **dissertation-engine**: LV results are the dissertation's Law 3 evidence

## Dependencies for Next Steps
- Integration with ternary-cell GC phase for population management
- Room-level competition modeling
- Fleet-level ecological simulation
