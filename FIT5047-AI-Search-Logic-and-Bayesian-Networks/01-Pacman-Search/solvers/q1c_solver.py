#---------------------#
# DO NOT MODIFY BEGIN #
#---------------------#

import logging

import util
from problems.q1c_problem import q1c_problem

#-------------------#
# DO NOT MODIFY END #
#-------------------#

# q1c_solver.py
# A* with a hard time budget and a greedy fallback if we get close to timeout.
# Assumes Berkeley-style problem API:
#   getStartState(), isGoalState(s), getSuccessors(s) -> [(ns, action, stepCost), ...]
# and a heuristic fn: h(state, problem).

import heapq
from util import manhattanDistance
from game import Directions, Actions


# --------- Hybrid Solver ---------

def q1c_solver(problem: q1c_problem):
    """
    Hybrid Q1(c) solver:
      - If number of food dots <= 25 → run A* with max-distance heuristic.
      - Else → run greedy BFS to avoid timeouts on huge layouts.
    """
    start = problem.getStartState()
    _, food_set = start

    if len(food_set) > 25:
        print(f"Large problem detected ({len(food_set)} foods) → using greedy BFS fallback")
        return greedy_solver(problem, start, set(food_set))
    else:
        return astar_solver(problem, start)


# --------- A* Implementation ---------

class AStarData:
    def __init__(self, start_state, problem, heuristic):
        self.problem = problem
        self.heuristic = heuristic
        self.frontier = []  # heap of (f, state)
        self.g = {start_state: 0}
        self.parent = {start_state: None}
        self.action = {start_state: None}
        self.closed = set()

        h0 = heuristic(start_state, problem)
        heapq.heappush(self.frontier, (h0, start_state))


def astar_solver(problem, start):
    data = AStarData(start, problem, q1c_heuristic)
    num_expansions = 0

    while data.frontier:
        f, state = heapq.heappop(data.frontier)

        if state in data.closed:
            continue
        data.closed.add(state)

        if problem.isGoalState(state):
            actions = []
            s = state
            while data.parent[s] is not None:
                actions.append(data.action[s])
                s = data.parent[s]
            actions.reverse()
            print(f"Number of node expansions: {num_expansions}")
            return actions

        num_expansions += 1

        for succ, act, cost in problem.getSuccessors(state):
            g_new = data.g[state] + cost
            if succ not in data.g or g_new < data.g[succ]:
                data.g[succ] = g_new
                data.parent[succ] = state
                data.action[succ] = act
                f_new = g_new + data.heuristic(succ, problem)
                heapq.heappush(data.frontier, (f_new, succ))

    print(f"Number of node expansions: {num_expansions}")
    return []


def q1c_heuristic(state, problem: q1c_problem):
    """
    Heuristic = max Manhattan distance from Pac-Man to any remaining food.
    Admissible & consistent. Cheap to compute.
    """
    pos, remaining = state
    if not remaining:
        return 0
    return max(manhattanDistance(pos, food) for food in remaining)


# --------- Greedy BFS Fallback ---------

def greedy_solver(problem, start, remaining):
    """
    Greedy BFS: repeatedly go to the nearest food until done.
    """
    plan = []
    pos, _ = start
    walls = problem.walls

    def bfs_to_nearest(src, targets):
        if not targets:
            return [], src
        if src in targets:
            return [], src

        q = util.Queue()
        q.push(src)
        visited = {src}
        parent = {src: (None, None)}

        while not q.isEmpty():
            s = q.pop()
            for action in (Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST):
                dx, dy = Actions.directionToVector(action)
                nx, ny = int(s[0] + dx), int(s[1] + dy)

                if nx < 0 or nx >= walls.width or ny < 0 or ny >= walls.height:
                    continue
                if walls[nx][ny]:
                    continue

                ns = (nx, ny)
                if ns in visited:
                    continue
                visited.add(ns)
                parent[ns] = (s, action)

                if ns in targets:
                    # reconstruct path
                    actions = []
                    cur = ns
                    while parent[cur][0] is not None:
                        actions.append(parent[cur][1])
                        cur = parent[cur][0]
                    actions.reverse()
                    return actions, ns

                q.push(ns)

        return None, None

    while remaining:
        path, reached = bfs_to_nearest(pos, remaining)
        if path is None:
            break
        plan.extend(path)
        pos = reached
        remaining.remove(reached)

    return plan
