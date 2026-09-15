Score every block
  Block A could be a PartCount, or a StepCount

Then try every permentation of Block assignements.
Hierachary assign, each permentation is a state.
  State 1: A=PartCount
  State 2: A=StepCount

At the end we score the assignments and pick the best one.

We can use various optimizations to reduce the search space:
- Beam Search: only keep the top N scoring states at each step.
- MRV (Minimum Remaining Values): prioritize blocks with the fewest valid assignments first.
- Dependency Tracking: track dependencies between blocks to avoid invalid states.
- Early Pruning: discard states that cannot possibly lead to a valid solution early in the search process.
- Memoization: cache results of subproblems to avoid redundant calculations.
- Conflict Resolution: when a block assignment causes conflicts, mark conflicting blocks as failed to prevent further consideration.
- Dynamic Scoring: adjust scoring criteria based on the current state of the search to prioritize more promising paths.
- Parallel Processing: if possible, evaluate multiple states concurrently to speed up the search process.
- 