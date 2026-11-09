#---------------------#
# DO NOT MODIFY BEGIN #
#---------------------#

import logging

import util
from problems.q1b_problem import q1b_problem

def q1b_solver(problem: q1b_problem):
    astarData = astar_initialise(problem)
    num_expansions = 0
    terminate = False
    while not terminate:
        num_expansions += 1
        terminate, result = astar_loop_body(problem, astarData)
    print(f'Number of node expansions: {num_expansions}')
    return result

#-------------------#
# DO NOT MODIFY END #
#-------------------#

class AStarData:
    """
    Minimal container for A* state.
    """
    def __init__(self, start, goals):
        self.start = start
        self.goals = goals                     # tuple of goal positions (food dots)
        self.frontier = util.PriorityQueue()   # holds (state, path) with priority f = g + h
        self.g = {start: 0}                    # best-known g-cost to each state
        self.closed = set()                    # expanded states


def astar_initialise(problem: q1b_problem):
    """
    Prepare A* frontier with the start node.
    """
    start = problem.getStartState()
    goals = getattr(problem, "foods", tuple())
    data = AStarData(start, goals)

    # seed frontier with start (empty path), f = h(start)
    h0 = astar_heuristic(start, goals)
    data.frontier.push((start, []), h0)
    return data


def astar_loop_body(problem: q1b_problem, astarData: AStarData):
    """
    Perform one A* iteration:
      - pop the best node
      - if goal, return full plan
      - else expand successors and push/update them
    Returns: (terminate: bool, result_actions: list|None)
    """
    if astarData.frontier.isEmpty():
        # no solution (unlikely on valid layouts)
        return True, []

    state, path = astarData.frontier.pop()

    # if already expanded, skip this node (no expansion performed)
    if state in astarData.closed:
        return False, None
    astarData.closed.add(state)

    # goal test: ANY food dot
    if problem.isGoalState(state):
        return True, path

    # expand successors
    for succ, action, step_cost in problem.getSuccessors(state):
        tentative_g = astarData.g[state] + step_cost
        if succ not in astarData.g or tentative_g < astarData.g[succ]:
            astarData.g[succ] = tentative_g
            h = astar_heuristic(succ, astarData.goals)
            f = tentative_g + h
            astarData.frontier.push((succ, path + [action]), f)

    return False, None


def astar_heuristic(current, goals):
    """
    Heuristic = min Manhattan distance to ANY goal (admissible & consistent).
    """
    if not goals:
        return 0
    cx, cy = current
    return min(abs(cx - gx) + abs(cy - gy) for (gx, gy) in goals)
