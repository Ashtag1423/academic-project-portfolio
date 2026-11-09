# Pacman: Heuristic and Adversarial Search

## Q1 — Heuristic search

- `solvers/q1a_solver.py`: A* with Manhattan distance for single-goal navigation.
- `solvers/q1b_solver.py`: A* with minimum Manhattan distance to the next goal.
- `solvers/q1c_solver.py`: A* with maximum Manhattan distance to remaining food; a greedy, nearest-food BFS approach is selected for larger instances (more than 25 food items in the submitted code).

The accompanying report discusses time constraints, heuristic tradeoffs and a Q1c assessment score of **6.96/10**. The implementation should be read as the submitted version; descriptions of deadline handling in the report may differ from individual code paths.

## Q2 — Adversarial multi-agent search

- `agents/q2Agent.py`: Pacman-vs-ghost agent with alpha-beta search, iterative deepening, state evaluation, maze-distance caching and tactical heuristics.
- The submitted report describes quiescence extensions, move ordering, aspiration windows and a transposition table, with local evaluation scores **around 17–18** for Q2. These are reported results, not independently reproduced here.

## Code scope

These four student-edited source files are extracted from a larger Berkeley-style Pacman assignment repository. They import `util`, `game`, `problems`, and other modules in the course-provided framework; running them requires that original framework and its layouts. The starter files and grading resources are intentionally not rehosted.
