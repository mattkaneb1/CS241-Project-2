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

        # Don't stand still
        pos = currentGameState.getPacmanPosition()
        if pos == newPos:
          return -float("inf")

        # How far was/is ghost
        minDistFromGhostNow  = float('inf')
        minDistFromGhostThen = float('inf')
        for g in range(len(newGhostStates)):
          minDistFromGhostNow  = min(minDistFromGhostNow,manhattanDistance(newPos,newGhostStates[g].getPosition()))
          minDistFromGhostThen = min(minDistFromGhostThen,manhattanDistance(pos,newGhostStates[g].getPosition()))
        
        # How far is closest food
        minDistFromFood = float('inf')
        for f in newFood.asList():
          d = manhattanDistance(newPos,f)
          minDistFromFood = min(minDistFromFood,d)

        # Do I eat a dot on this turn
        dotEaten = int(len(newFood.asList()) < len(currentGameState.getFood().asList()))

        # If the ghost was close, make the move that leaves me the farthest away
        if minDistFromGhostThen < 3:
          return minDistFromGhostNow
        
        # If not, do moves that eat dots or get you closer to the nearest dot
        else:
          return dotEaten + 1/float(minDistFromFood)

        # Performs pretty well in regular maze(Really bad in openClassic)
        """
        pos = currentGameState.getPacmanPosition()
        if pos == newPos:
          return -0.5
        
        minDistFromGhost = float('inf')
        for g in range(len(newGhostStates)):
            if newScaredTimes[g] == 0:
              minDistFromGhost = min(minDistFromGhost,manhattanDistance(newPos,newGhostStates[g].getPosition()))

        if minDistFromGhost < 3 :
          print("Danger" + str(minDistFromGhost))
          return minDistFromGhost

        else:
          print(successorGameState.getScore())
          return successorGameState.getScore()
        """

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def isTerminal(self, agent, state, depth):
        if agent == 0:
            return (state.isWin() or state.isLose()) or self.depth == depth
        else:
            return state.isWin() or state.isLose()

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

        def pacMove(state, depth):
            if self.isTerminal(0, state, depth+1): #self.terminalTest(state) or self.cutoffTest(depth+1):
                return self.evaluationFunction(state)
            legalActions = state.getLegalActions()
            currMax = float('-inf')
            for action in legalActions:
                successorState = state.generateSuccessor(0, action)
                currMax = max(currMax, ghostMove(1, successorState, depth+1)) #first ghost has next move
            return currMax

        def ghostMove(ghostNum, state, depth):
            if self.isTerminal(ghostNum, state, depth): #self.terminalTest(state): #no cutoff check because depth only updated after all ghosts make move
                return self.evaluationFunction(state)
            legalActions = state.getLegalActions(ghostNum)
            currMin = float('inf')
            for action in legalActions:
                successorState = state.generateSuccessor(ghostNum, action)
                if (ghostNum + 1) % gameState.getNumAgents() != 0: #if not last ghost, the next ghost moves next
                    currMin = min(currMin, ghostMove(ghostNum + 1, successorState, depth))
                else: #if last ghost, pacman has next move
                    currMin = min(currMin, pacMove(successorState, depth))
            return currMin

        legalActions = gameState.getLegalActions() #start of game, pacman has first move
        maximum = float('-inf')
        bestAction = None
        for action in legalActions:
            successorState = gameState.generateSuccessor(0, action)
            currVal = ghostMove(1, successorState, 0) #first ghost has next move
            if currVal > maximum:
                maximum = currVal
                bestAction = action
        return bestAction

        """
        if we're at the max level, take the action that maximizes chances
        if gameState is terminal
            return evaluationFunction(gameState)
        get actions for max
        for action in actions
            generate the successor from resulting action
            get the max(currentMax, minLevel(ghost, depth, nextAgent))
        return max
        
        if we're at min level, take action that minimizes reward
        if gameState is terminal
            return evaluationFunciton(gamestate)
        get actions for min
        for action in actions
            generate successor from resulting action
            if next agent is ghost
                get the min(currentMin, minLevel(successor, depth, ghost)
            if next agent is pacman
                get min(currentMin, maxLevel(successor, depth)
        return min
        
        at the root
        generate actions for pacman
        for action in actions
            get successor from action
            currValue = minLevel(successor, depth, ghostIndex=1)
            if currValue > max
                max = currValue
                bestAction = action
        return bestAction
        """


        """
        m = self.miniMax(gameState, 0, -1) #call minimax for pacman
        d = {} #dictionary that maps actions with values
        for action in gameState.getLegalActions(0):
            d[action] = self.miniMax(gameState.generateSuccessor(0, action), -1, 1)

        v = max(d.values()) #find the max value
        for m in d.keys():
            if d[m] == v:
                return m #return action associated with max value

    def miniMax(self, gameState, depth, agent):
        if self.terminalTest(gameState):
            return self.evaluationFunction(gameState) #gameState.getScore()
        elif self.cutoffTest(depth):
            return self.evaluationFunction(gameState) #gameState.getScore()
        else:
            nextAgent = (agent + 1) % gameState.getNumAgents()

            # If Pacman
            if agent == 0:
                pacmanMoves = gameState.getLegalActions(agent)
                scores = []
                for move in pacmanMoves:
                    scores.append(self.miniMax(gameState.generateSuccessor(agent, move), depth + 1, nextAgent))
                return max(scores)

            else:
                ghostMoves = gameState.getLegalActions(agent)
                scores = []
                for move in ghostMoves:
                    if nextAgent == 0:
                        scores.append(self.miniMax(gameState.generateSuccessor(agent, move), depth, nextAgent))
                    else:
                        scores.append(self.miniMax(gameState.generateSuccessor(agent, move), depth, nextAgent))
                return min(scores)
        """

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def isTerminal(self, agent, state, depth):
        if agent == 0:
            return (state.isWin() or state.isLose()) or self.depth == depth
        else:
            return state.isWin() or state.isLose()

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        legalActions = gameState.getLegalActions() 
        maximum = float('-inf')
        bestAction = None
        for action in legalActions:
            successorState = gameState.generateSuccessor(0, action)
            currVal = self.minValue(successorState,float('-inf'),float('inf'),1,0)
            if currVal > maximum:
                maximum = currVal
                bestAction = action
        return bestAction

    def minValue(self,state,alpha,beta,agent,depth):

      if self.isTerminal(agent,state,depth):
        return self.evaluationFunction(state)

      v = float('inf')

      nextAgent = (agent + 1) % state.getNumAgents()
      for action in state.getLegalActions(agent):
        successorState = state.generateSuccessor(agent, action)

        if nextAgent == 0:
          v = min(v,self.maxValue(successorState,alpha,beta,nextAgent,depth+1))
        else:
          v = min(v,self.minValue(successorState,alpha,beta,nextAgent,depth))

        if v < alpha:
          return v

        beta = min(beta,v)

      return v

    def maxValue(self,state,alpha,beta,agent,depth):

      if self.isTerminal(agent,state,depth):
        return self.evaluationFunction(state)

      v = float('-inf')

      nextAgent = (agent + 1) % state.getNumAgents()

      for action in state.getLegalActions(agent):
        successorState = state.generateSuccessor(agent, action)

        v = max(v,self.minValue(successorState,alpha,beta,nextAgent,depth))

        if v > beta:
          return v

        alpha = max(alpha,v)

      return v



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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

