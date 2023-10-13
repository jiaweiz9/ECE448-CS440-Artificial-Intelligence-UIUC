# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
import heapq

# Search should return the path and the number of states explored.
# The path should be a list of MazeState objects that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (astar)
# You may need to slight change your previous search functions in MP2 since this is 3-d maze


def search(maze, searchMethod):
    return {
        "astar": astar,
    }.get(searchMethod, [])(maze)


# TODO: VI
def astar(maze):
    starting_state = maze.get_start()
    visited_states = {starting_state: (None, 0)}

    # The frontier is a priority queue
    # You can pop from the queue using "heapq.heappop(frontier)"
    # You can push onto the queue using "heapq.heappush(frontier, state)"
    # NOTE: states are ordered because the __lt__ method of AbstractState is implemented
    frontier = []
    heapq.heappush(frontier, starting_state)
    #print(starting_state)
    # TODO(III): implement the rest of the best first search algorithm
    # HINTS:
    #   - add new states to the frontier by calling state.get_neighbors()
    #   - check whether you've finished the search by calling state.is_goal()
    #       - then call backtrack(visited_states, state)...
    # Your code here ---------------
    while len(frontier) != 0 :
        frontier_first = heapq.heappop(frontier)
        #print("pop: ", frontier_first)
        #frontier_first.dist_from_start = visited_states.get(frontier_first)[1]
        if frontier_first.dist_from_start > visited_states.get(frontier_first)[1]:
            continue
        if frontier_first.is_goal():
            #frontier_first.get_neighbors()
            return backtrack(visited_states, frontier_first)
        potential_nbrs = frontier_first.get_neighbors()
        #print(potential_nbrs, "\n")
        for nbr in potential_nbrs:
            if visited_states.get(nbr) == None or nbr.dist_from_start < visited_states.get(nbr)[1]:
            #if visited_states.get(nbr) == None:
                visited_states.update({nbr: (frontier_first, nbr.dist_from_start)})
                heapq.heappush(frontier, nbr)
                #print("push:", nbr)
            # elif nbr.dist_from_start < visited_states.get(nbr)[1]:
            #     visited_states.update({nbr: (frontier_first, nbr.dist_from_start)})
                
        #print("\n")
    # ------------------------------

    # if you do not find the goal return an empty list
    return None


# Go backwards through the pointers in visited_states until you reach the starting state
# NOTE: the parent of the starting state is None
# TODO: VI
def backtrack(visited_states, current_state):
    path = []
    # Your code here ---------------
    # print("goal:", goal_state)
    # print(visited_states)
    # print(visited_states.get(goal_state)[0])
    while visited_states.get(current_state)[0] is not None:
        path.insert(0, current_state)
        print("current_state:", current_state, ", dist_from_start=", current_state.dist_from_start)
        current_state = visited_states[current_state][0]
    path.insert(0, current_state)
    print(path)
    # ------------------------------
    return path
