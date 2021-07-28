
#Importing necessary modules, frameworks

import numpy as np
import matplotlib.pyplot as plt
import heapq as pq
import copy as cp
import time
import json
import queue


# Global variables

number_of_nodes = 0
initial_state = np.array([[]])   
goal = np.array([[]])
state_status = {}
distance = {}

actions = {
    'UP' : [-1,0],
    'DOWN' : [1,0],
    'LEFT' : [0,-1],
    'RIGHT' : [0,1]
}


# Offline mode calculation

states = np.array([
                    [['2', '5', '4'],
                     ['7', '6', '1'],
                     ['8', '3', '-']],

                    [['2', '5', '4'],
                     ['7', '6', '1'],
                     ['8', '-', '3']],

                    [['2', '5', '4'],
                     ['7', '6', '1'],
                     ['-', '8', '3']],

                    [['2', '5', '4'],
                     ['-', '6', '1'],
                     ['7', '8', '3']],

                    [['2', '5', '4'],
                     ['6', '-', '1'],
                     ['7', '8', '3']],

                    [['2', '5', '4'],
                     ['6', '1', '-'],
                     ['7', '8', '3']],

                    [['2', '5', '4'],
                     ['6', '1', '3'],
                     ['7', '8', '-']],

                    [['2', '5', '4'],
                     ['6', '1', '3'],
                     ['7', '-', '8']],

                    [['2', '5', '4'],
                     ['6', '1', '3'],
                     ['-', '7', '8']],

                    [['2', '5', '4'],
                     ['-', '1', '3'],
                     ['6', '7', '8']],

                    [['2', '5', '4'],
                     ['1', '-', '3'],
                     ['6', '7', '8']],

                    [['2', '5', '4'],
                     ['1', '3', '-'],
                     ['6', '7', '8']],

                    [['2', '5', '-'],
                     ['1', '3', '4'],
                     ['6', '7', '8']],

                    [['2', '-', '5'],
                     ['1', '3', '4'],
                     ['6', '7', '8']],

                    [['-', '2', '5'],
                     ['1', '3', '4'],
                     ['6', '7', '8']],

                    [['1', '2', '5'],
                     ['-', '3', '4'],
                     ['6', '7', '8']],

                    [['1', '2', '5'],
                     ['3', '-', '4'],
                     ['6', '7', '8']],

                    [['1', '2', '5'],
                     ['3', '4', '-'],
                     ['6', '7', '8']],

                    [['1', '2', '-'],
                     ['3', '4', '5'],
                     ['6', '7', '8']],

                    [['1', '-', '2'],
                     ['3', '4', '5'],
                     ['6', '7', '8']]
                ])

states = np.flip(states,0)

goal = np.array([['-','1','2'],
                 ['3','4','5'],
                 ['6','7','8']])


stats_for_plots = {
    
    'States' : states.tolist(),
    'Optimal_cost' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
    'BFS' : {
        'clock_time' : [],
        'nodes_generated' : [],
        'path_cost' : []
    },
    'UCS' : {
        'clock_time' : [],
        'nodes_generated' : [],
        'path_cost' : []
    },
    'DLS' : {
        'clock_time' : [],
        'nodes_generated' : [],
        'path_cost' : []
    },
    'IDS' : {
        'clock_time' : [],
        'nodes_generated' : [],
        'path_cost' : []
    },
    'GBFS' : {
        'misplaced_tiles' : {
            'clock_time' : [],
            'nodes_generated' : [],
            'path_cost' : []
        },
        'manhattan_distance' : {
            'clock_time' : [],
            'nodes_generated' : [],
            'path_cost' : []
        }
    },
    'A_star' : {
        'misplaced_tiles' : {
            'clock_time' : [],
            'nodes_generated' : [],
            'path_cost' : []
        },
        'manhattan_distance' : {
            'clock_time' : [],
            'nodes_generated' : [],
            'path_cost' : []
        }
    }
}



# necessary methods

def is_transition_valid(shape, x, y):
    if x < 0 or x >= shape[0]:
        return False
    
    if y < 0 or y >= shape[1]:
        return False
    
    return True

def no_of_misplaced_tiles(state_1, state_2):
    misplaced_tiles = np.sum(state_1 != state_2)
    
    if not (np.where(state_1 == '-') == np.where(state_2 == '-')):
        misplaced_tiles -= 1
    return misplaced_tiles

def calculate_manhattan_distance(x1, y1, x2, y2):
    return abs(x1-x2) + abs(y1-y2)

def find_pos(state,char):
    pos = np.where(state == char)
    return int(pos[0]), int(pos[1])

def total_manhattan_distance(state_1, state_2):
    total = 0
    
    for i in range(state_1.shape[0]):
        for j in range(state_1.shape[1]):
            if(state_1[i][j] != '-'):
                x1,y1 = i,j
                x2,y2 = find_pos(state_2,state_1[i][j])
                total += calculate_manhattan_distance(x1,y1,x2,y2)
    return total

def generate_goal(n):
    board = []
    for i in range(n):
        chars = []
        for j in range(i*n, (i+1)*n):
            if not j: chars.append('-')
            else: chars.append(str(j))
        board.append(chars)
        
    return np.array(board)

def clear_dicts():
    state_status.clear()
    distance.clear()
    
def print_path(file, node):
    if node.parent is None:
        file.write(str(node.state) + "\n")
        return
    print_path(file,node.parent)
    file.write("\n")
    file.write("Action : "+ node.action + "\n\n")
    file.write(str(node.state) + "\n")
    
def print_solution(algo, solution, clock_time):
    file = open(algo + "_solution.txt", "w+")
    file.write("Solution for " + algo + " :\n\n")
    if solution is not None:
        if solution != 'cutoff':
            file.write("Initial board configuaration : \n")
            file.write(str(initial_state) + "\n\n")
            file.write("Goal state : \n")
            file.write(str(solution.state) + "\n\n")
            
            file.write("Algorithm took " + str(clock_time) + " seconds.\n")
            file.write("Solution cost : " + str(solution.g_cost) + "\n")
            file.write("Solution depth : " + str(solution.depth) + "\n")
            file.write("Number of nodes generated : " + str(number_of_nodes) + "\n\n")
            file.write("Solution path : \n\n")
            print_path(file,solution)
        else:
            file.write("cutoff\n")

    else:
        file.write("No solution\n")
    
    file.write("\n")
    file.close()


# data structure for node(Search tree)

class node:
    
    def __init__(self, state, parent, action, depth, g_cost = 0, h_cost = 0, algo = "UCS"):
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = depth
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.algo = algo
        global number_of_nodes
        number_of_nodes += 1
        
    def __lt__(self, other):
        if self.algo == "GBFS": return self.h_cost < other.h_cost # for greedy best first search
        elif self.algo == "A_star": return self.g_cost + self.h_cost < other.g_cost + other.h_cost # for A* search
        return self.g_cost < other.g_cost # for uniformed cost search
        
    def is_goal_state(self, goal):
        return np.array_equal(self.state, goal)
    
    def blank_space_pos(self):
        blank_pos = np.where(self.state == '-')
        return int(blank_pos[0]), int(blank_pos[1])
    
    def set_heuristic_cost(self, goal, heuristic):
        if heuristic == "misplaced_tiles":
            self.h_cost = no_of_misplaced_tiles(self.state, goal)
            
        elif heuristic == "manhattan_distance":
            self.h_cost = total_manhattan_distance(self.state, goal)
 



# Breadth-first search

def BFS():
    
    clear_dicts()
    
    algo = "BFS"
    root_node = node(state = cp.deepcopy(initial_state), parent = None, action = None, depth = 0, g_cost = 0, h_cost = 0, algo = algo)
    
    if root_node.is_goal_state(goal):
        return root_node
    
    frontier = queue.Queue(maxsize = 0)  # maxsize of zero ‘0’ means a infinite queue
    frontier.put(root_node)
    state_status[str(root_node.state)] = 'frontier'
    
    while True:
        if frontier.empty():
            return None
        cur_node = frontier.get()
        state_status[str(cur_node.state)] = 'explored'
        
        for action in actions:
            pos_x, pos_y = cur_node.blank_space_pos()
            
            new_pos_x = pos_x + actions[action][0]
            new_pos_y = pos_y + actions[action][1]
            
            if is_transition_valid(cur_node.state.shape, new_pos_x, new_pos_y):
                temp_state = cp.deepcopy(cur_node.state)
                temp_state[pos_x][pos_y], temp_state[new_pos_x][new_pos_y] = temp_state[new_pos_x][new_pos_y], temp_state[pos_x][pos_y]
                child_node = node(state = cp.deepcopy(temp_state), parent = cur_node, action = action, depth = cur_node.depth + 1, g_cost = cur_node.g_cost + 1, h_cost = 0, algo = "BFS")
                
                if not(str(child_node.state) in state_status):
                    if child_node.is_goal_state(goal):
                        return child_node
                    frontier.put(child_node)
                    state_status[str(child_node.state)] = 'frontier'
    return None                
    



# Uniform-cost search

def UCS():
    
    clear_dicts()
    
    algo = "UCS"
    root_node = node(state = cp.deepcopy(initial_state), parent = None, action = None, depth = 0, g_cost = 0, h_cost = 0, algo = algo)
    
    frontier = []
    pq.heapify(frontier) 
    pq.heappush(frontier,root_node)
    
    state_status[str(root_node.state)] = 'frontier'
    distance[str(root_node.state)] = root_node.g_cost
    
    while True:
        if not len(frontier):
            return None
        cur_node = pq.heappop(frontier)
        state_status[str(cur_node.state)] = 'explored'
        
        if cur_node.is_goal_state(goal):
            return cur_node
        
        for action in actions:
            pos_x, pos_y = cur_node.blank_space_pos()
            
            new_pos_x = pos_x + actions[action][0]
            new_pos_y = pos_y + actions[action][1]
            
            if is_transition_valid(cur_node.state.shape, new_pos_x, new_pos_y):
                temp_state = cp.deepcopy(cur_node.state)
                temp_state[pos_x][pos_y], temp_state[new_pos_x][new_pos_y] = temp_state[new_pos_x][new_pos_y], temp_state[pos_x][pos_y]
                child_node = node(state = cp.deepcopy(temp_state), parent = cur_node, action = action, depth = cur_node.depth + 1, g_cost = cur_node.g_cost + 1, h_cost = 0, algo = "UCS")
    
                
                if not(str(child_node.state) in state_status):
                    pq.heappush(frontier,child_node)
                    distance[str(child_node.state)] = child_node.g_cost
                    state_status[str(child_node.state)] = 'frontier'
                    
                elif distance[str(child_node.state)] > child_node.g_cost:
                    distance[str(child_node.state)] = child_node.g_cost
                    pq.heappush(frontier,child_node)
                    
    return None                
    
    



# Depth-first search

def recursive_DLS(cur_node, limit, algo):
    
    if cur_node.is_goal_state(goal):
        return cur_node
    
    elif not limit:
        return 'cutoff'
    
    else:

        cutoff_flag = False
        
        for action in actions:
            pos_x, pos_y = cur_node.blank_space_pos()
            
            new_pos_x = pos_x + actions[action][0]
            new_pos_y = pos_y + actions[action][1]
            
            if is_transition_valid(cur_node.state.shape, new_pos_x, new_pos_y):
                temp_state = cp.deepcopy(cur_node.state)
                temp_state[pos_x][pos_y], temp_state[new_pos_x][new_pos_y] = temp_state[new_pos_x][new_pos_y], temp_state[pos_x][pos_y]
                child_node = node(state = cp.deepcopy(temp_state), parent = cur_node, action = action, depth = cur_node.depth + 1, g_cost = cur_node.g_cost + 1, h_cost = 0, algo = algo)
                
                if not(str(child_node.state) in state_status) or state_status[str(child_node.state)] < limit:
                    
                    state_status[str(child_node.state)] = limit
                    result = recursive_DLS(child_node, limit - 1, algo)
                    
                    if result == 'cutoff':
                        cutoff_flag = True
                    elif result is not None:
                        return result
        
        if cutoff_flag:
            return 'cutoff'
        
        return None

    
def DLS(limit = 26, algo = "DLS"):
    
    clear_dicts()
    
    root_node = node(state = cp.deepcopy(initial_state), parent = None, action = None, depth = 0, g_cost = 0, h_cost = 0, algo = algo)
    
    state_status[str(root_node.state)] = limit
    result = recursive_DLS(root_node, limit, algo)
    if str(root_node.state) in state_status:
        del state_status[str(root_node.state)]
    
    return result



# Iterative deepening depth-first search

def IDS():
    algo = "IDS"

    for limit in range(30):
        result = DLS(limit, "IDS")
        print(limit, result)
        if result != 'cutoff':
            return result
        



# Greedy best-first search

def GBFS(heuristic = "manhattan_distance"):
    
    clear_dicts()
    
    algo = "GBFS"
    root_node = node(state = cp.deepcopy(initial_state), parent = None, action = None, depth = 0, g_cost = 0, h_cost = 0, algo = algo)

    
    frontier = []
    pq.heapify(frontier) 
    pq.heappush(frontier,root_node)
    
    state_status[str(root_node.state)] = 'frontier'
    #distance[str(root_node.state)] = root_node.h_cost
    
    while True:
        if not len(frontier):
            return None
        cur_node = pq.heappop(frontier)
        state_status[str(cur_node.state)] = 'explored'
        
        if cur_node.is_goal_state(goal):
            return cur_node
        
        for action in actions:
            pos_x, pos_y = cur_node.blank_space_pos()
            
            new_pos_x = pos_x + actions[action][0]
            new_pos_y = pos_y + actions[action][1]
            
            if is_transition_valid(cur_node.state.shape, new_pos_x, new_pos_y):
                temp_state = cp.deepcopy(cur_node.state)
                temp_state[pos_x][pos_y], temp_state[new_pos_x][new_pos_y] = temp_state[new_pos_x][new_pos_y], temp_state[pos_x][pos_y]
    
                child_node = node(state = cp.deepcopy(temp_state), parent = cur_node, action = action, depth = cur_node.depth + 1, g_cost = cur_node.g_cost + 1, h_cost = 0, algo = "GBFS")
                child_node.set_heuristic_cost(goal, heuristic)
                
                if not(str(child_node.state) in state_status):
                    pq.heappush(frontier,child_node)
                    #distance[str(child_node.state)] = child_node.h_cost
                    state_status[str(child_node.state)] = 'frontier'
                    
#                 elif distance[str(child_node.state)] > child_node.h_cost:
#                     distance[str(child_node.state)] = child_node.h_cost
#                     pq.heappush(frontier,child_node)
                    
                    
    return None     



# A* search

def A_star(heuristic = "manhattan_distance"):
    
    clear_dicts()
    
    algo = "A_star"
    root_node = node(state = cp.deepcopy(initial_state), parent = None, action = None, depth = 0, g_cost = 0, h_cost = 0, algo = algo)
    
    frontier = []
    pq.heapify(frontier) 
    pq.heappush(frontier,root_node)
    
    state_status[str(root_node.state)] = 'frontier'
    distance[str(root_node.state)] = root_node.g_cost + root_node.h_cost 
    
    while True:
        if not len(frontier):
            return None
        cur_node = pq.heappop(frontier)
        state_status[str(cur_node.state)] = 'explored'
        
        if cur_node.is_goal_state(goal):
            return cur_node
        
        for action in actions:
            pos_x, pos_y = cur_node.blank_space_pos()
            
            new_pos_x = pos_x + actions[action][0]
            new_pos_y = pos_y + actions[action][1]
            
            if is_transition_valid(cur_node.state.shape, new_pos_x, new_pos_y):
                temp_state = cp.deepcopy(cur_node.state)
                temp_state[pos_x][pos_y], temp_state[new_pos_x][new_pos_y] = temp_state[new_pos_x][new_pos_y], temp_state[pos_x][pos_y]
                
                child_node = node(state = cp.deepcopy(temp_state), parent = cur_node, action = action, depth = cur_node.depth + 1, g_cost = cur_node.g_cost + 1, h_cost = 0, algo = "A_star")
                child_node.set_heuristic_cost(goal, heuristic)
                
                if not(str(child_node.state) in state_status):
                    pq.heappush(frontier,child_node)
                    distance[str(child_node.state)] = child_node.g_cost + child_node.h_cost
                    state_status[str(child_node.state)] = 'frontier'
                    
                elif distance[str(child_node.state)] > child_node.g_cost + child_node.h_cost:
                    distance[str(child_node.state)] = child_node.g_cost + child_node.h_cost
                    pq.heappush(frontier,child_node)
                    
                    
    return None  


# Offline mode calculation

def initialize_stats_for_plots():
    algorithms = ['BFS', 'UCS', 'DLS', 'IDS', 'GBFS', 'A_star']
    heuristics = ['misplaced_tiles', 'manhattan_distance']
    
    for algo in algorithms:
        
        if algo == 'GBFS' or algo == 'A_star':
            for heuristic in heuristics:
                stats_for_plots[algo][heuristic]['clock_time'].clear()
                stats_for_plots[algo][heuristic]['nodes_generated'].clear()
                stats_for_plots[algo][heuristic]['path_cost'].clear()
            
        else:
            stats_for_plots[algo]['clock_time'].clear()
            stats_for_plots[algo]['nodes_generated'].clear()
            stats_for_plots[algo]['path_cost'].clear()

            
# Offline mode calculation

def generate_BFS_statistics():
    
    # Generate statistics using BFS for some states of 8-puzzle problem that requires
    # varying steps, such as 1, 2, ..., 20, to reach the solution. 

    for i in range(states.shape[0]):

        global number_of_nodes
        global initial_state

        algo = 'BFS'
        number_of_nodes = 0
        initial_state = np.array(states[i])

        start_time = time.time()
        solution = BFS()
        clock_time = time.time() - start_time

        stats_for_plots[algo]['clock_time'].append(clock_time)
        stats_for_plots[algo]['nodes_generated'].append(number_of_nodes)
        stats_for_plots[algo]['path_cost'].append(solution.g_cost)

        print(algo + ": Solution cost : " + str(solution.g_cost))
        print(algo + ": --- %s seconds ---" % clock_time)
        print("Number of nodes generated : " + str(number_of_nodes))
        print("")

        clear_dicts()



# Offline mode calculation

def generate_UCS_statistics():

    # Generate statistics using UCS for some states of 8-puzzle problem that requires
    # varying steps, such as 1, 2, ..., 20, to reach the solution. 

    for i in range(states.shape[0]):

        global number_of_nodes
        global initial_state

        algo = 'UCS'
        number_of_nodes = 0
        initial_state = np.array(states[i])

        start_time = time.time()
        solution = UCS()
        clock_time = time.time() - start_time

        stats_for_plots[algo]['clock_time'].append(clock_time)
        stats_for_plots[algo]['nodes_generated'].append(number_of_nodes)
        stats_for_plots[algo]['path_cost'].append(solution.g_cost)

        print(algo + ": Solution cost : " + str(solution.g_cost))
        print(algo + ": --- %s seconds ---" % clock_time)
        print("Number of nodes generated : " + str(number_of_nodes))
        print("")

        clear_dicts()



# Offline mode calculation

def generate_DLS_statistics():

    # Generate statistics using DLS for some states of 8-puzzle problem that requires
    # varying steps, such as 1, 2, ..., 20, to reach the solution. 

    for i in range(states.shape[0]):

        global number_of_nodes
        global initial_state

        algo = 'DLS'
        number_of_nodes = 0
        initial_state = np.array(states[i])

        start_time = time.time()
        solution = DLS()
        clock_time = time.time() - start_time

        stats_for_plots[algo]['clock_time'].append(clock_time)
        stats_for_plots[algo]['nodes_generated'].append(number_of_nodes)
        stats_for_plots[algo]['path_cost'].append(solution.g_cost)

        print(algo + ": Solution cost : " + str(solution.g_cost))
        print(algo + ": --- %s seconds ---" % clock_time)
        print("Number of nodes generated : " + str(number_of_nodes))
        print("")

        clear_dicts()



# Offline mode calculation

def generate_IDS_statistics():
    
    # Generate statistics using IDS for some states of 8-puzzle problem that requires
    # varying steps, such as 1, 2, ..., 20, to reach the solution. 

    for i in range(states.shape[0]):

        global number_of_nodes
        global initial_state

        algo = 'IDS'
        number_of_nodes = 0
        initial_state = np.array(states[i])

        start_time = time.time()
        solution = IDS()
        clock_time = time.time() - start_time

        stats_for_plots[algo]['clock_time'].append(clock_time)
        stats_for_plots[algo]['nodes_generated'].append(number_of_nodes)
        stats_for_plots[algo]['path_cost'].append(solution.g_cost)

        print(algo + ": Solution cost : " + str(solution.g_cost))
        print(algo + ": --- %s seconds ---" % clock_time)
        print("Number of nodes generated : " + str(number_of_nodes))
        print("")

        clear_dicts()



# Offline mode calculation

def generate_GBFS_with_misplaced_tiles_statistics():

    # Generate statistics using GBFS(Heuristic = Number of misplaced tiles) for some states of 8-puzzle problem that requires
    # varying steps, such as 1, 2, ..., 20, to reach the solution. 

    for i in range(states.shape[0]):

        global number_of_nodes
        global initial_state

        heuristic = "misplaced_tiles"  
        algo = "GBFS"
        number_of_nodes = 0
        initial_state = np.array(states[i])

        start_time = time.time()
        solution = GBFS(heuristic)
        clock_time = time.time() - start_time

        stats_for_plots[algo][heuristic]['clock_time'].append(clock_time)
        stats_for_plots[algo][heuristic]['nodes_generated'].append(number_of_nodes)
        stats_for_plots[algo][heuristic]['path_cost'].append(solution.g_cost)

        print(algo+" : " + heuristic + " : Solution cost : " + str(solution.g_cost))
        print(algo+" : " + heuristic + " : --- %s seconds ---" % clock_time)
        print("Number of nodes generated : " + str(number_of_nodes))
        print("")

        clear_dicts()



# Offline mode calculation

def generate_GBFS_with_manhattan_distance_statistics():

    # Generate statistics using GBFS(Heuristic = Manhattan Distance) for some states of 8-puzzle problem that requires
    # varying steps, such as 1, 2, ..., 20, to reach the solution. 

    for i in range(states.shape[0]):

        global number_of_nodes
        global initial_state

        heuristic = "manhattan_distance"    
        algo = "GBFS"
        number_of_nodes = 0
        initial_state = np.array(states[i])

        start_time = time.time()
        solution = GBFS(heuristic)
        clock_time = time.time() - start_time

        stats_for_plots[algo][heuristic]['clock_time'].append(clock_time)
        stats_for_plots[algo][heuristic]['nodes_generated'].append(number_of_nodes)
        stats_for_plots[algo][heuristic]['path_cost'].append(solution.g_cost)

        print(algo+" : " + heuristic + " : Solution cost : " + str(solution.g_cost))
        print(algo+" : " + heuristic + " : --- %s seconds ---" % clock_time)
        print("Number of nodes generated : " + str(number_of_nodes))
        print("")

        clear_dicts()



# Offline mode calculation

def generate_A_star_with_misplaced_tiles_statistics():

    # Generate statistics using A_star(Heuristic = Number of misplaced tiles) for some states of 8-puzzle problem that requires
    # varying steps, such as 1, 2, ..., 20, to reach the solution. 

    for i in range(states.shape[0]):

        global number_of_nodes
        global initial_state

        heuristic = "misplaced_tiles"  
        algo = "A_star"
        number_of_nodes = 0
        initial_state = np.array(states[i])

        start_time = time.time()
        solution = A_star(heuristic)
        clock_time = time.time() - start_time

        stats_for_plots[algo][heuristic]['clock_time'].append(clock_time)
        stats_for_plots[algo][heuristic]['nodes_generated'].append(number_of_nodes)
        stats_for_plots[algo][heuristic]['path_cost'].append(solution.g_cost)

        print(algo+" : " + heuristic + " : Solution cost : " + str(solution.g_cost))
        print(algo+" : " + heuristic + " : --- %s seconds ---" % clock_time)
        print("Number of nodes generated : " + str(number_of_nodes))
        print("")

        clear_dicts()



# Offline mode calculation

def generate_A_star_with_manhattan_distance_statistics():

    # Generate statistics using A_star(Heuristic = Manhattan Distance) for some states of 8-puzzle problem that requires
    # varying steps, such as 1, 2, ..., 20, to reach the solution. 

    for i in range(states.shape[0]):

        global number_of_nodes
        global initial_state

        heuristic = "manhattan_distance"    
        algo = "A_star"
        number_of_nodes = 0
        initial_state = np.array(states[i])

        start_time = time.time()
        solution = A_star(heuristic)
        clock_time = time.time() - start_time

        stats_for_plots[algo][heuristic]['clock_time'].append(clock_time)
        stats_for_plots[algo][heuristic]['nodes_generated'].append(number_of_nodes)
        stats_for_plots[algo][heuristic]['path_cost'].append(solution.g_cost)

        print(algo+" : " + heuristic + " : Solution cost : " + str(solution.g_cost))
        print(algo+" : " + heuristic + " : --- %s seconds ---" % clock_time)
        print("Number of nodes generated : " + str(number_of_nodes))
        print("")

        clear_dicts()


# Offline mode calculation

def write_statistics_to_json():
    stats_jason = json.dumps(stats_for_plots)
    f = open("statistics.json","w")
    f.write(stats_jason)
    f.close()




while True:
    
    # selecting mode option #

    mode = 1
    while True:
        print("Enter 1 for Testing mode")
        print("Enter 2 for Offline mode")
        mode = int(input("Select a mode : "))
        if mode == 1 or mode == 2: 
            break

    if mode == 1:
        # testing mode #

        print("Enter 3 for 3-puzzle")
        print("Enter 8 for 8-puzzle")
        print("Enter 15 for 15-puzzle")

        n = 8
        sqrt_n = int(np.sqrt(n+1))
        while True:
            n = int(input("Enter the value of n for n-puzzle: "))
            sqrt_n = int(np.sqrt(n+1))
            if (sqrt_n * sqrt_n) == n+1: break

        board  = []

        print("Enter the numbers for the puzzle separated by spaces and enter - for blank space. \n")

        for i in range(sqrt_n):

            chars = []
            while True:
                row = input("Enter row " + str(i+1) + ": ")
                chars = row.split()
                if len(chars) == sqrt_n: break
                else: print("Enter exactly " + str(sqrt_n) + " numbers.")

            board.append(chars)

        initial_state = np.array(board)        
        print("Initial board configuation : ")
        print(initial_state)

        goal = generate_goal(sqrt_n)
        print("Goal state configuation : ")
        print(goal)

        print("\nPlease pick a algorithm or all.")
        print("Enter 1 for Breadth first search(BFS).\nEnter 2 for Uniform cost search(UCS).\nEnter 3 for depth limited search(DLS)")
        print("Enter 4 for Iterative deepening depth first search(IDS).\nEnter 5 for Greedy best first search(GBFS)")
        print("Enter 6 for A* search.\nEnter 7 for all algorithms.")

        option = 1
        while True:
            option = int(input("Enter a number : "))
            if option >= 1 and option <= 7: break

        heuristic_option = 1
        if option >= 5 and option <= 7:
            print('\nChoose a heuristic for GBFS and/or A* search.')
            print('Enter 1 for "The number of misplaced tiles".')
            print('Enter 2 for "The Manhattan distance".')

            while True:
                heuristic_option = int(input("Enter a number : "))
                if heuristic_option >= 1 and heuristic_option <= 2: break

        print("")

        if option in [1,7]:
            algo = "BFS"
            number_of_nodes = 0

            start_time = time.time()
            solution = BFS()
            clock_time = time.time() - start_time
            
            if solution is not None:
                if solution != 'cutoff':
                    print(algo + ": Solution cost : " + str(solution.g_cost))
                    print(algo + ": --- %s seconds ---" % clock_time)
                    print("Number of nodes generated : " + str(number_of_nodes))
                    print("")
                else:
                    print("cutoff\n")

            else:
                print("No solution\n")

            print_solution(algo, solution, clock_time)
            clear_dicts()

        if option in [2,7]:
            algo = "UCS"
            number_of_nodes = 0
            
            start_time = time.time()
            solution = UCS()
            clock_time = time.time() - start_time
            
            if solution is not None:
                if solution != 'cutoff':
                    print(algo + ": Solution cost : " + str(solution.g_cost))
                    print(algo + ": --- %s seconds ---" % clock_time)
                    print("Number of nodes generated : " + str(number_of_nodes))
                    print("")
                else:
                    print("cutoff\n")

            else:
                print("No solution\n")

            print_solution(algo, solution, clock_time)
            clear_dicts()


        if option in [5,7]:
            if heuristic_option == 1: heuristic = "misplaced_tiles"
            else: heuristic = "manhattan_distance"    

            algo = "GBFS"
            number_of_nodes = 0
            
            start_time = time.time()
            solution = GBFS(heuristic)
            clock_time = time.time() - start_time
            
            if solution is not None:
                if solution != 'cutoff':
                    print(algo + ": Solution cost : " + str(solution.g_cost))
                    print(algo + ": --- %s seconds ---" % clock_time)
                    print("Number of nodes generated : " + str(number_of_nodes))
                    print("")
                else:
                    print("cutoff\n")

            else:
                print("No solution\n")

            print_solution(algo, solution, clock_time)
            clear_dicts()

        if option in [6,7]:
            if heuristic_option == 1: heuristic = "misplaced_tiles"
            else: heuristic = "manhattan_distance"

            algo = "A_star"
            number_of_nodes = 0

            start_time = time.time()
            solution = A_star(heuristic)
            clock_time = time.time() - start_time
            
            if solution is not None:
                if solution != 'cutoff':
                    print(algo + ": Solution cost : " + str(solution.g_cost))
                    print(algo + ": --- %s seconds ---" % clock_time)
                    print("Number of nodes generated : " + str(number_of_nodes))
                    print("")
                else:
                    print("cutoff\n")

            else:
                print("No solution\n")

            print_solution(algo, solution, clock_time)
            clear_dicts()
            
        if option in [3,7]:
            algo = "DLS"
            number_of_nodes = 0

            start_time = time.time()
            solution = DLS()
            clock_time = time.time() - start_time
            
            if solution is not None:
                if solution != 'cutoff':
                    print(algo + ": Solution cost : " + str(solution.g_cost))
                    print(algo + ": --- %s seconds ---" % clock_time)
                    print("Number of nodes generated : " + str(number_of_nodes))
                    print("")
                else:
                    print("cutoff\n")

            else:
                print("No solution\n")
            
            print_solution(algo, solution, clock_time)
            clear_dicts()

        if option in [4,7]:
            algo = "IDS"
            number_of_nodes = 0

            start_time = time.time()
            solution = IDS()
            clock_time = time.time() - start_time
            
            if solution is not None:
                if solution != 'cutoff':
                    print(algo + ": Solution cost : " + str(solution.g_cost))
                    print(algo + ": --- %s seconds ---" % clock_time)
                    print("Number of nodes generated : " + str(number_of_nodes))
                    print("")
                else:
                    print("cutoff\n")

            else:
                print("No solution\n")
            
            print_solution(algo, solution, clock_time)
            clear_dicts()
        
        print("Do you want to continue from the beginning?")
    
    elif mode == 2:
        
        # offline mode #

        # initialize_stats_for_plots()
        # generate_BFS_statistics()
        # generate_UCS_statistics()
        # generate_DLS_statistics()
        # generate_IDS_statistics()
        # generate_GBFS_with_misplaced_tiles_statistics()
        # generate_GBFS_with_manhattan_distance_statistics()
        # generate_A_star_with_misplaced_tiles_statistics()
        # generate_A_star_with_manhattan_distance_statistics()       
        # write_statistics_to_json()


        with open("statistics.json") as f_in:
            statistics = json.load(f_in)
        
        metrics = ['clock_time', 'nodes_generated', 'path_cost']
        plt.rcParams["figure.figsize"] = (10,16)

        plt_index = 1
        for metric in metrics:
            plt.subplot(3,1, plt_index)
            plt.plot(statistics['Optimal_cost'], statistics['BFS'][metric], color="#FF4933", label = "BFS")
            plt.plot(statistics['Optimal_cost'], statistics['UCS'][metric], color = "#FFFE33", label = "UCS")
            plt.plot(statistics['Optimal_cost'], statistics['DLS'][metric], color="#B9FF33", label = "DLS")
            plt.plot(statistics['Optimal_cost'], statistics['IDS'][metric], color = "#33FF73", label = "IDS")
            plt.plot(statistics['Optimal_cost'], statistics['GBFS']['misplaced_tiles'][metric], color = "#33FFEB", label = "GBFS(misplaced_tiles)")
            plt.plot(statistics['Optimal_cost'], statistics['GBFS']['manhattan_distance'][metric], color = "#33ADFF", label = "GBFS(manhattan_distance)")
            plt.plot(statistics['Optimal_cost'], statistics['A_star']['misplaced_tiles'][metric], color = "#5734F3", label = "A_star(misplaced_tiles)")
            plt.plot(statistics['Optimal_cost'], statistics['A_star']['manhattan_distance'][metric], color = "#AE34F3", label = "A_star(manhattan_distance)")

            # plt.xlim(-32,32)
            # plt.ylim(-10,10)
            plt.xlabel('Optimal_cost')
            plt.ylabel(metric)
            plt.legend(prop={'size': 8})
            plt.title(metric + ' vs Optimal cost')
            plt_index += 1
        
        plt.tight_layout(pad = 6)
        plt.grid()
        plt.show()
        
    again = 0    
    while True:
        again = int(input("If yes enter 1, otherwise enter 0.\n"))
        if again in [0,1]: break
        
    if not again: break
