# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """




    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(gameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return gameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        super().__init__()
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """



    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        def minimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            numAgents = gameState.getNumAgents()

            if agentIndex == 0:
                return max_value(agentIndex, depth, gameState)

            else:
                return min_value(agentIndex, depth, gameState)

        def max_value(agentIndex, depth, gameState):
            legalActions = gameState.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(gameState)

            bestValue = float('-inf')

            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                value = minimax(1, depth, successorState)
                bestValue = max(bestValue, value)

            return bestValue

        def min_value(agentIndex, depth, gameState):
            legalActions = gameState.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(gameState)

            bestValue = float('inf')

            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:
                    value = minimax(0, depth + 1, successorState)
                else:
                    value = minimax(agentIndex + 1, depth, successorState)
                bestValue = min(bestValue, value)

            return bestValue

        bestAction = None
        bestScore = float('-inf')
        legalActions = gameState.getLegalActions(0)

        for action in legalActions:
            successorState = gameState.generateSuccessor(0, action)
            score = minimax(1, 0, successorState)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    minimax agent with alpha-beta pruning
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        def alphaBeta(agentIndex, depth, gameState, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            if agentIndex == 0:
                return max_value(agentIndex, depth, gameState, alpha, beta)

            else:
                return min_value(agentIndex, depth, gameState, alpha, beta)

        def max_value(agentIndex, depth, gameState, alpha, beta):
                legalActions = gameState.getLegalActions(agentIndex)
                if not legalActions:
                    return self.evaluationFunction(gameState)

                bestValue = float('-inf')

                for action in legalActions:
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    value = alphaBeta(1, depth, successorState, alpha, beta)
                    bestValue = max(bestValue, value)

                    alpha = max(alpha, bestValue)

                    if beta <= alpha:
                        break

                return bestValue

        def min_value(agentIndex, depth, gameState, alpha, beta):
                legalActions = gameState.getLegalActions(agentIndex)
                if not legalActions:
                    return self.evaluationFunction(gameState)

                bestValue = float('inf')

                for action in legalActions:
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    if agentIndex == gameState.getNumAgents() - 1:
                        value = alphaBeta(0, depth + 1, successorState, alpha, beta)
                    else:
                        value = alphaBeta(agentIndex + 1, depth, successorState, alpha, beta)

                    bestValue = min(bestValue, value)

                    beta = min(beta, bestValue)

                    if beta <= alpha:
                        break

                return bestValue

        bestAction = None
        bestScore = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        legalActions = gameState.getLegalActions(0)

        if not legalActions:
            return None

        for action in legalActions:
            successorState = gameState.generateSuccessor(0, action)
            score = alphaBeta(1, 0, successorState, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action

            alpha = max(alpha, bestScore)

        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        super().__init__(evalFn, depth)
        self.last_positions = []
        self.no_food_steps = 0
        self.previous_food_distance = 0

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction.

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        best_action, utility = self.expectimax(gameState, depth=0, agentIndex=0)
        return best_action

    def expectimax(self, gameState, depth, agentIndex):
        """
        The main expectimax function that recursively calculates the best action.

        Arguments:
        gameState -- the current game state
        depth -- the current depth in the tree
        agentIndex -- the index of the current agent (0 for Pacman, 1+ for ghosts)

        Returns:
        A tuple of (action, utility) where:
        - action is the best action for the agent at the current node
        - utility is the expectimax value of the state
        """

        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return None, self.evaluationFunction(gameState)

        num_agents = gameState.getNumAgents()

        if agentIndex == 0:
            return self.max_value(gameState, depth, agentIndex)

        else:
            return self.exp_value(gameState, depth, agentIndex)

    def max_value(self, gameState, depth, agentIndex):
        """
        Maximizing function for Pacman (agentIndex == 0).
        """
        best_action = None
        max_utility = float("-inf")

        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            _, utility = self.expectimax(successor, depth, agentIndex + 1)

            if utility > max_utility:
                max_utility = utility
                best_action = action

        return best_action, max_utility

    def exp_value(self, gameState, depth, agentIndex):
        """
        Expectation function for ghosts (agentIndex > 0).
        """
        actions = gameState.getLegalActions(agentIndex)
        total_utility = 0

        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)

            if agentIndex == gameState.getNumAgents() - 1:
                _, utility = self.expectimax(successor, depth + 1, 0)
            else:
                _, utility = self.expectimax(successor, depth, agentIndex + 1)

            total_utility += utility

        avg_utility = total_utility / len(actions) if actions else 0
        return None, avg_utility

    def evaluationFunction(self, gameState):
        """
        An evaluation function that considers:
        - Food distance and amount
        - Proximity to power-ups (shields, freezers, intangible objects)
        - Ghost proximity and state (shielded/frozen or intangible)
        """
        pacman_pos = gameState.getPacmanPosition()
        food_list = gameState.getFood().asList()
        ghost_states = gameState.getGhostStates()
        score = gameState.getScore()

        utility = score

        # ** Food and capsule proximity **
        if food_list:
            closest_food_dist = min([util.manhattanDistance(pacman_pos, food) for food in food_list])

            # if self.previous_food_distance is not None and closest_food_dist > self.previous_food_distance:
            #     # Apply penalty if distance to food has increased
            #      utility -= (abs(closest_food_dist - self.previous_food_distance)) * 5

            self.previous_food_distance = closest_food_dist

            if len(food_list) < 10:
                # Get the distance to the closest food pellet
                utility += 300 / (closest_food_dist + 1)

            else:
                utility += 5 / (closest_food_dist + 1)
                # Reset idle counter if Pacman collected food
                if pacman_pos in food_list:
                    utility += 5000
                    self.no_food_steps = 0
                else:
                    self.no_food_steps += 1

                # Apply penalty if idle for too long without collecting food
            if self.no_food_steps > 15:
                utility -= self.no_food_steps * 10  # Increase penalty with time spent idle

        if gameState.isShieldEaten():
            utility += 100

        if gameState.isFreezerEaten():
            utility += 100

        # ** Power-ups proximity **
        if gameState.getShields():
            closest_shield_dist = min([util.manhattanDistance(pacman_pos, shield) for shield in gameState.getShields()])
            utility += 15 / (closest_shield_dist + 1)

        if gameState.getFreezers():
            closest_freezer_dist = min([util.manhattanDistance(pacman_pos, freezer) for freezer in gameState.getFreezers()])
            utility += 15 / (closest_freezer_dist + 1)

        if gameState.getIntangibleObJ():
            closest_intangible_dist = min(
                [util.manhattanDistance(pacman_pos, intangible) for intangible in gameState.getIntangibleObJ()])
            utility += 1000 / (closest_intangible_dist + 1)

        # ** Ghost proximity **
        for ghost in ghost_states:
            ghost_distance = util.manhattanDistance(pacman_pos, ghost.getPosition())

            if ghost_distance > 0:
                if gameState.isShielded or gameState.isFrozen or ghost.scaredTimer > 0:
                    # Reduced threat if shielded or frozen
                    utility += 25 / (ghost_distance + 1)
                elif gameState.getIntangible:
                    # Pass-through mode
                    utility += 5 / (ghost_distance + 1)
                else:
                    # Normal threat level
                    utility -= 20 / ghost_distance
        # ** Oscillation detection **
        # Track last few positions and check for repeated oscillation patterns
        self.last_positions.append(pacman_pos)
        if len(self.last_positions) > 16:
            self.last_positions.pop(0)

        # Check if Pacman is oscillating between two or three positions
        if len(set(self.last_positions)) <= 4:
            utility -= 20  # Penalty for oscillating pattern

        return utility





# Abbreviation
# better = betterEvaluationFunction

