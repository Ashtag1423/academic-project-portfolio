#---------------------#
# DO NOT MODIFY BEGIN #
#---------------------#

import logging

import util
from problems.q1a_problem import q1a_problem

def q1a_solver(problem: q1a_problem):
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
    Container for A* search state.
    """
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
        self.frontier = util.PriorityQueue()   # holds (state, path) with priority = g + h
        self.g = {start: 0}                    # best-known g-cost to each state
        self.parent = {start: (None, None)}    # state -> (prev_state, action)
        self.closed = set()                    # expanded states


def astar_initialise(problem: q1a_problem):
    """
    Initialise A* with start and goal from the problem.
    """
    start = problem.getStartState()
    # q1a_problem caches the single-dot goal
    goal = getattr(problem, "goal", start)
    data = AStarData(start, goal)

    # push the start node (empty path) with f = h(start, goal)
    h0 = astar_heuristic(start, goal)
    data.frontier.push((start, []), h0)
    return data


def astar_loop_body(problem: q1a_problem, data: AStarData):
    """
    Perform one iteration of A*:
      - pop the best node
      - if goal, reconstruct and terminate
      - else expand successors and push/update them
    Returns: (terminate: bool, result_actions: list|None)
    """
    # if no nodes left, terminate (no solution)
    if data.frontier.isEmpty():
        print("pathLength: 0")
        return True, []

    state, path = data.frontier.pop()

    # skip if already expanded (lazy duplicate detection)
    if state in data.closed:
        return False, None
    data.closed.add(state)

    # goal test
    if problem.isGoalState(state):
        print(f"pathLength: {len(path)}")
        return True, path

    # expand successors
    for succ, action, step_cost in problem.getSuccessors(state):
        tentative_g = data.g[state] + step_cost
        if succ not in data.g or tentative_g < data.g[succ]:
            data.g[succ] = tentative_g
            h = astar_heuristic(succ, data.goal)
            f = tentative_g + h
            data.parent[succ] = (state, action)
            data.frontier.push((succ, path + [action]), f)

    return False, None


def astar_heuristic(current, goal):
    """
    Manhattan distance — admissible and consistent on a 4-connected grid with unit costs.
    """
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1])
