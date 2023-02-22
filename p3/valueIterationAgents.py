# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
from collections import defaultdict
from typing import Dict

def computeQValue(
        mdp: mdp.MarkovDecisionProcess,
        values: Dict, discount: float,
        state, action):
    """
    $ Run a step of Q-value iteration: it computes Q[k+1](s, a) given
    the last computed values v[k].
    """
    return sum(
        prob * (mdp.getReward(state, action, nextState) + discount * values[nextState])
        for nextState, prob in mdp.getTransitionStatesAndProbs(state, action)
    )


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = defaultdict(float) # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        states = mdp.getStates()
        qValues = list()
        prevValues = defaultdict(int)
        currValues = self.values

        for i in range(self.iterations):
            prevValues, currValues = currValues, prevValues
            for state in states:
                actions = mdp.getPossibleActions(state)
                for action in actions:
                    qValue = computeQValue(mdp, prevValues, self.discount, state, action)
                    qValues.append(qValue)
                if not qValues:
                    currValues[state] = 0
                else:
                    currValues[state] = max(qValues)
                qValues.clear()

        self.values = currValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        discount = self.discount
        values = self.values
        sumValues = list()

        for nextState, prob in mdp.getTransitionStatesAndProbs(state, action):
            val = prob*(mdp.getReward(state, action, nextState) + discount*values[nextState])
            sumValues.append(val)
        
        return sum(sumValues)


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        bestAction = None
        
        if mdp.isTerminal(state):
            return None
        else:
            actions = mdp.getPossibleActions(state)
            for action in actions:
                val = self.getQValue(state, action)
                if bestAction is None:
                    bestAction = action
                #If the qValue is greater than the best action
                elif val > self.getQValue(state, bestAction): 
                        bestAction = action
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
