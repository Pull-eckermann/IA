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
from pacman import GameState
import math

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        score = 0 #Começa o escore em zero
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        currentPos = currentGameState.getPacmanPosition()   #Posição atual
        newPos = successorGameState.getPacmanPosition()     #Posição nova

        newFoods = successorGameState.getFood().asList()    #Lista de comidas no novo estado
        currentFoods = currentGameState.getFood().asList()  #Lista de comidas no estado atual

        newGPos = successorGameState.getGhostPositions()    #Lista de posição dos fantasmas

        for i in range(0,len(newGPos)):
            if manhattanDistance(newPos, newGPos[i]) <= 4:
                score -= 5; #Se estiver muito perto de um fantasma, foge

        #Se na proxima posição a comida estiver mais próxima, ganha pontos
        b_current_distance = self.melhorDistanciaFoods(currentFoods, currentPos)
        b_new_distance = self.melhorDistanciaFoods(newFoods, newPos)
        if b_new_distance < b_current_distance:
            score += 3
        #Se na próxima posição tiver menos comida, recebe prioridade
        if len(newFoods) < len(currentFoods):
            score += 1

        return score

    def melhorDistanciaFoods(self, foods : list, pos : tuple):
        b_distance = 1000
        for food in foods:
            distance = manhattanDistance(pos, food)
            if distance < b_distance:
                b_distance = distance
        
        return b_distance


def scoreEvaluationFunction(currentGameState: GameState):
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
    only partially specified, aGameStatend designed to be extended.  Agent (game.py)
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

    def getAction(self, gameState: GameState):
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
        legalMoves = gameState.getLegalActions()    #Lista de Movimentos legais
        #Cria uma lista com os estados sucessores
        successors = [gameState.generateSuccessor(self.index, move) for move in legalMoves]
        #De acordo com minmax, pontua cada estado sucessor
        scores = [self.valor_minmax(successor, self.index + 1, self.depth) for successor in successors]
        #Pega o melhor score, acha o indice e retorna o movimento correspondente
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

    def valor_minmax(self, gameState : GameState, agente : int, depth : int):
        if gameState.isWin() or gameState.isLose() or (depth == 0):
            return self.evaluationFunction(gameState)
        if agente == 0:
            return self.max_value(gameState, agente, depth)
        else: 
            return self.min_value(gameState, agente, depth)


    def min_value(self, gameState : GameState, agente : int, depth : int):
        if agente ==  gameState.getNumAgents() - 1:
            depth -= 1
            newAgente = 0
        else:
            newAgente = agente + 1

        legalMoves = gameState.getLegalActions(agente)
        successors = [gameState.generateSuccessor(agente, move) for move in legalMoves]
        scores = [self.valor_minmax(successor, newAgente, depth) for successor in successors]
        
        return min(scores)

    def max_value(self, gameState : GameState, agente : int, depth : int):
        legalMoves = gameState.getLegalActions(agente)
        successors = [gameState.generateSuccessor(agente, move) for move in legalMoves]
        scores = [self.valor_minmax(successor, agente+1, depth) for successor in successors]
        
        return max(scores)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -math.inf
        betha = math.inf
        _, bestMove = self.valor_minmax(gameState, self.index, self.depth, alpha, betha)
        return bestMove

    def valor_minmax(self, gameState : GameState, agente : int, depth : int, alpha, betha):
        if gameState.isWin() or gameState.isLose() or (depth == 0):
            return self.evaluationFunction(gameState), None
        if agente == 0:
            return self.max_value(gameState, agente, depth, alpha, betha)
        else: 
            return self.min_value(gameState, agente, depth, alpha, betha)


    def min_value(self, gameState : GameState, agente : int, depth : int, alpha, betha):
        if agente ==  gameState.getNumAgents() - 1:
            depth -= 1
            newAgente = 0
        else:
            newAgente = agente + 1

        worstScore = math.inf
        worstMove = None

        legalMoves = gameState.getLegalActions(agente)
        for move in legalMoves:
            successor = gameState.generateSuccessor(agente, move)
            score, _ = self.valor_minmax(successor, newAgente, depth, alpha, betha)
            if score < alpha:
                return score, move
            if score < worstScore:
                worstScore = score
                worstMove = move
                if score < betha:
                    betha = score
        
        return worstScore, worstMove

    def max_value(self, gameState : GameState, agente : int, depth : int, alpha, betha):
        bestScore = -math.inf
        bestMove = None

        legalMoves = gameState.getLegalActions(agente)
        for move in legalMoves:
            successor = gameState.generateSuccessor(agente, move)
            score, _ = self.valor_minmax(successor, agente+1, depth, alpha, betha)
            if score > betha:
                return score, move
            if score > bestScore:
                bestScore = score
                bestMove = move
                if score > alpha:
                    alpha = score
        
        return bestScore, bestMove

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions()    #Lista de Movimentos legais
        #Cria uma lista com os estados sucessores
        successors = [gameState.generateSuccessor(self.index, move) for move in legalMoves]
        #De acordo com minmax, pontua cada estado sucessor
        scores = [self.valor_expecmax(successor, self.index + 1, self.depth) for successor in successors]
        #Pega o melhor score, acha o indice e retorna o movimento correspondente
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

    def valor_expecmax(self, gameState : GameState, agente : int, depth : int):
        if gameState.isWin() or gameState.isLose() or (depth == 0):
            return self.evaluationFunction(gameState)
        if agente == 0:
            return self.max_value(gameState, agente, depth)
        else: 
            return self.expec_value(gameState, agente, depth)


    def expec_value(self, gameState : GameState, agente : int, depth : int):
        if agente ==  gameState.getNumAgents() - 1:
            depth -= 1
            newAgente = 0
        else:
            newAgente = agente + 1

        score = 0
        legalMoves = gameState.getLegalActions(agente)
        successors = [gameState.generateSuccessor(agente, move) for move in legalMoves]
        for successor in successors:
            prob = 1 / len(legalMoves)
            score += prob * self.valor_expecmax(successor, newAgente, depth)
        
        return score

    def max_value(self, gameState : GameState, agente : int, depth : int):
        legalMoves = gameState.getLegalActions(agente)
        successors = [gameState.generateSuccessor(agente, move) for move in legalMoves]
        scores = [self.valor_expecmax(successor, agente+1, depth) for successor in successors]
        
        return max(scores)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacPos = currentGameState.getPacmanPosition()   #Posição do Pacman
    ghostPos = currentGameState.getGhostPositions() #Posição dos Fantasmas
    foods = currentGameState.getFood().asList()     #Lista de comidas
    numFoods = len(foods)                           #Numero de comidas

    bestDisfood = 1                 
    currentScore = currentGameState.getScore()
    #Coloca em uma lista as distâncias entre o pacman e as comidas
    food_distances = [manhattanDistance(pacPos, food_position) for food_position in foods]

    if numFoods > 0:
        bestDisfood = min(food_distances)

    #Se a distancia entre pacman e um ghost for pequena,
    #a melhor distancia pra uma comida desvaloriza muito
    for ghost in ghostPos:
        ghostDis = manhattanDistance(pacPos, ghost)
        if ghostDis <= 3:
            bestDisfood = 10000

    caracteristicas = ((1.0/bestDisfood), currentScore, numFoods)
    pesos = (5, 100, -50)

    return sum([caracteristica * peso for caracteristica, peso in zip(caracteristicas, pesos)])

# Abbreviation
better = betterEvaluationFunction
