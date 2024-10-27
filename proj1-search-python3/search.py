# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
import random

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """

        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """

        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    "*** YOUR CODE HERE ***"
    stack = util.Stack()
    currentPosition = problem.getStartState()
    stack.push((currentPosition, []))

    visited = set()

    while not stack.isEmpty():
        state, path = stack.pop()

        if problem.isGoalState(state):
            return path

        if state not in visited:
            visited.add(state)

            for successor, action, stepCost in problem.getSuccessors(state):
                if successor not in visited:
                    stack.push((successor, path + [action]))


    return []

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    visited = set()
    queue = util.Queue()
    currentPosition = problem.getStartState()
    queue.push((currentPosition, []))

    while not queue.isEmpty():
        state, path = queue.pop()
        if problem.isGoalState(state):
            return path

        if state not in visited:
            visited.add(state)

            for successor, action, stepCost in problem.getSuccessors(state):
                if successor not in visited:
                    queue.push((successor, path + [action]))

    return []
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):

    """
    Perform Real-Time A* search on the FoodSearchProblem.
    """
    startState = problem.getStartState()
    foodList = problem.getStartState()[1].asList()
    if not foodList:
        return []  # No food to collect, return empty action list

    actions = []
    currentState = startState

    while True:
        path = aStarSearchHelper(currentState, problem, heuristic)
        if not path:
            break  # No more actions possible

        actions.extend(path)
        # Update the state after taking the actions
        for action in path:
            successors = problem.getSuccessors(currentState)
            # Update currentState based on the action taken
            for successor in successors:
                if successor[1] == action:  # match action
                    currentState = successor[0]
                    break

        # Update food list
        foodList = currentState[1].asList()
        if not foodList:
            break  # All food collected

    return actions

def aStarSearchHelper(state, problem, heuristic=nullHeuristic):
    """
    Performs the A* search from the current state and returns the path to the goal.
    """
    from util import PriorityQueue
    visited = set()
    queue = PriorityQueue()
    queue.push((state, []), 0)

    while not queue.isEmpty():
        currentState, path = queue.pop()
        if currentState in visited:
            continue

        visited.add(currentState)
        if problem.isGoalState(currentState):
            return path

        for successor, action, cost in problem.getSuccessors(currentState):
            newPath = path + [action]
            priority = len(newPath) + heuristic(successor, problem)
            queue.push((successor, newPath), priority)

    return []

    # priorityQueue = util.PriorityQueue()
    # currentState = problem.getStartState()
    #
    # priorityQueue.push((currentState, [], 0), 0 + heuristic(currentState, problem))
    #
    # visited = set()
    #
    # while not priorityQueue.isEmpty():
    #     state, path, costUntilNow = priorityQueue.pop()
    #
    #     if problem.isGoalState(state):
    #         return path
    #
    #     if state not in visited:
    #         visited.add(state)
    #
    #         for successor, action, stepCost in problem.getSuccessors(state):
    #             if successor not in visited:
    #                 newCost = costUntilNow + stepCost
    #                 priority = newCost + heuristic(successor, problem)
    #                 priorityQueue.push((successor, path + [action], newCost), priority)
    #
    # return []

def randomSearch(problem):
    """

    """
    currentPosition = problem.getStartState()
    solution = []
    while( not (problem.isGoalState(currentPosition))):
        successors = problem.getSuccessors(currentPosition)
        randomIndex = random.randint(0, len(successors)-1)
        nextPosition = successors[randomIndex]
        solution.append(nextPosition[1])
        currentPosition = nextPosition[0]
    print(f'Steps {solution}')
    return solution

def allcorners(problem):
    currentPosition = problem.getStartState()

    priorityQueue = util.PriorityQueue()





# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
