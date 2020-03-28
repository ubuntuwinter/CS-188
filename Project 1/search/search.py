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
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    from util import Stack  # 使用栈数据结构
    closeList = []  # 存储已经展开的结点
    stack = Stack()
    stack.push((problem.getStartState(), []))  # 压入初始结点，包括问题的状态以及到达该状态的路径
    while True:
        nowState = stack.pop()  # 弹出栈底状态（结点）
        if nowState[0] in closeList:
            continue  # 如果已经展开过，则跳过
        else:
            closeList.append(nowState[0])  # 如果没有展开过，则添加到closeList
        if problem.isGoalState(nowState[0]):
            actions = nowState[1]  # 如果弹出的是目标状态，则取出路径，结束循环
            break
        for state, action, cost in problem.getSuccessors(nowState[0]):  # 展开
            newActions = nowState[1][:]  # 复制一下
            newActions.append(action)  # 添加
            stack.push((state, newActions))  # 压入展开的结点
    return actions
    # util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue  # 使用队列数据结构，其他都与dfs相同
    closeList = []
    queue = Queue()
    queue.push((problem.getStartState(), []))
    while True:
        nowState = queue.pop()
        if nowState[0] in closeList:
            continue
        else:
            closeList.append(nowState[0])
        if problem.isGoalState(nowState[0]):
            actions = nowState[1]
            break
        for state, action, cost in problem.getSuccessors(nowState[0]):
            newActions = nowState[1][:]
            newActions.append(action)
            queue.push((state, newActions))
    return actions
    # util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue  # 使用优先队列，其余都与bfs相同
    closeList = []
    queue = PriorityQueue()
    queue.push((problem.getStartState(), []), 0)
    while True:
        nowState = queue.pop()
        if nowState[0] in closeList:
            continue
        else:
            closeList.append(nowState[0])
        if problem.isGoalState(nowState[0]):
            actions = nowState[1]
            break
        for state, action, cost in problem.getSuccessors(nowState[0]):
            newActions = nowState[1][:]
            newActions.append(action)
            queue.push((state, newActions), problem.getCostOfActions(newActions))  # 代价作为优先
    return actions
    # util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue  # 引入启发函数，其余与ucs相同。
    closeList = []
    queue = PriorityQueue()
    queue.push((problem.getStartState(), []), 0)
    while True:
        nowState = queue.pop()
        if nowState[0] in closeList:
            continue
        else:
            closeList.append(nowState[0])
        if problem.isGoalState(nowState[0]):
            actions = nowState[1]
            break
        for state, action, cost in problem.getSuccessors(nowState[0]):
            newActions = nowState[1][:]
            newActions.append(action)
            # g(n) + h(n)
            queue.push((state, newActions), problem.getCostOfActions(newActions) + heuristic(state, problem))
    return actions
    # util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
