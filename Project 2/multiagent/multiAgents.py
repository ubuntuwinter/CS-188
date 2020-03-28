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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        value = successorGameState.getScore()
        newGhostPositions = successorGameState.getGhostPositions()
        if len(newGhostPositions) != 0:
            nearestGhostDistance = min(
                [manhattanDistance(newPos, newGhostPosition) for newGhostPosition in newGhostPositions])
            if nearestGhostDistance < 2:
                value -= 10
        newFoodPositions = newFood.asList()
        if len(newFoodPositions) != 0:
            nearestFoodDistance = min(
                [manhattanDistance(newPos, newFoodPosition) for newFoodPosition in newFoodPositions])
            value += 4 / nearestFoodDistance
        return value


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


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
        self.index = 0  # Pacman is always agent index 0
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minimax(state, depth, agentIndex):
            value = -999999999
            best_action = None
            for action in state.getLegalActions(agentIndex):
                now_value = minValue(state.generateSuccessor(0, action), depth, agentIndex + 1)
                if now_value >= value:
                    value = now_value
                    best_action = action
            return best_action

        def maxValue(state, depth, agentIndex):
            if state.isLose() or state.isWin() or depth == 0:
                return self.evaluationFunction(state)
            else:
                v = -999999999
                for action in state.getLegalActions(agentIndex):
                    v = max(v, minValue(state.generateSuccessor(agentIndex, action), depth, agentIndex + 1))
                return v

        def minValue(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            else:
                v = 999999999
                for action in state.getLegalActions(agentIndex):
                    if agentIndex == state.getNumAgents() - 1:
                        v = min(v, maxValue(state.generateSuccessor(agentIndex, action), depth - 1, 0))
                    else:
                        v = min(v, minValue(state.generateSuccessor(agentIndex, action), depth, agentIndex + 1))
                return v

        return minimax(gameState, self.depth, 0)
        # util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def minimax(state, alpha, beta, depth, agentIndex):
            value = -999999999
            best_action = None
            for action in state.getLegalActions(agentIndex):
                now_value = minValue(state.generateSuccessor(0, action), alpha, beta, depth, agentIndex + 1)
                if now_value >= value:
                    value = now_value
                    best_action = action
                if value > beta:
                    return best_action
                alpha = max(alpha, value)
            return best_action

        def maxValue(state, alpha, beta, depth, agentIndex):
            if state.isLose() or state.isWin() or depth == 0:
                return self.evaluationFunction(state)
            else:
                v = -999999999
                for action in state.getLegalActions(agentIndex):
                    v = max(v,
                            minValue(state.generateSuccessor(agentIndex, action), alpha, beta, depth, agentIndex + 1))
                    if v > beta:
                        return v
                    alpha = max(alpha, v)
                return v

        def minValue(state, alpha, beta, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            else:
                v = 999999999
                for action in state.getLegalActions(agentIndex):
                    if agentIndex == state.getNumAgents() - 1:
                        v = min(v, maxValue(state.generateSuccessor(agentIndex, action), alpha, beta, depth - 1, 0))
                    else:
                        v = min(v, minValue(state.generateSuccessor(agentIndex, action), alpha, beta, depth,
                                            agentIndex + 1))
                    if v < alpha:
                        return v
                    beta = min(beta, v)
                return v

        return minimax(gameState, -999999999, 999999999, self.depth, 0)
        # util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax(state, depth, agentIndex):
            value = -999999999
            best_action = None
            for action in state.getLegalActions(agentIndex):
                now_value = minValue(state.generateSuccessor(0, action), depth, agentIndex + 1)
                if now_value >= value:
                    value = now_value
                    best_action = action
            return best_action

        def maxValue(state, depth, agentIndex):
            if state.isLose() or state.isWin() or depth == 0:
                return self.evaluationFunction(state)
            else:
                v = -999999999
                for action in state.getLegalActions(agentIndex):
                    v = max(v, minValue(state.generateSuccessor(agentIndex, action), depth, agentIndex + 1))
                return v

        def minValue(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            else:
                v = 0
                l = len(state.getLegalActions(agentIndex))
                for action in state.getLegalActions(agentIndex):
                    if agentIndex == state.getNumAgents() - 1:
                        v += maxValue(state.generateSuccessor(agentIndex, action), depth - 1, 0) / l
                    else:
                        v += minValue(state.generateSuccessor(agentIndex, action), depth, agentIndex + 1) / l
                return v

        return expectimax(gameState, self.depth, 0)

    # util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    value = currentGameState.getScore()

    newGhostPositions = currentGameState.getGhostPositions()
    if len(newGhostPositions) != 0:
        nearestGhostDistance = 999999999
        nearestGhostIndex = -1
        for i in range(0, currentGameState.getNumAgents() - 1):
            ghostDistance = manhattanDistance(newPos, newGhostPositions[i])
            if ghostDistance < nearestGhostDistance:
                nearestGhostDistance = ghostDistance
                nearestGhostIndex = i
        if newScaredTimes[nearestGhostIndex] == 0 and nearestGhostDistance < 2:
            value -= 10
        if newScaredTimes[nearestGhostIndex] >= nearestGhostDistance:
            value += 20

    newCapsulesPositions = currentGameState.getCapsules()
    if len(newCapsulesPositions) != 0:
        nearestCapsulesPostion = min(
            [manhattanDistance(newPos, newCapsulesPosition) for newCapsulesPosition in newCapsulesPositions])
        if nearestCapsulesPostion < 2:
            value += 15

    newFoodPositions = newFood.asList()
    if len(newFoodPositions) != 0:
        nearestFoodDistance = min(
            [manhattanDistance(newPos, newFoodPosition) for newFoodPosition in newFoodPositions])
        value += 4 / nearestFoodDistance
    return value
    # util.raiseNotDefined()

    # Abbreviation


better = betterEvaluationFunction
