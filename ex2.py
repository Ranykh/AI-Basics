from copy import deepcopy
from collections import deque
from math import inf
import time
import itertools


ids = ["111111111", "222222222"]

class OptimalPirateAgent:
    def __init__(self, initial):

        # Map
        self.grid = tuple(tuple(sub) for sub in initial.get("map"))

        # Map size
        self.gridRowsNum = len(self.grid)
        self.gridColumnsNum = len(self.grid[0])

        # Treasures
        self.treasures_info = [(treasure_key, treasure.get("possible_locations"), treasure.get("prob_change_location"))
                               for treasure_key, treasure in initial.get("treasures").items()]
        # Marines
        self.marines_info = [(marine_key, marine.get("path")) for marine_key, marine in
                             initial.get("marine_ships").items()]
        # Time
        self.horizon = int(initial.get("turns to go"))
        # Walls
        original_map = initial["map"]
        rows = len(original_map)
        cols = len(original_map[0]) if rows > 0 else 0
        expanded_rows = rows + 2
        expanded_cols = cols + 2
        check_boundaries = [['W' for _ in range(expanded_cols)] for _ in range(expanded_rows)]
        for i in range(rows):
            for j in range(cols):
                check_boundaries[i + 1][j + 1] = original_map[i][j]
        self.check_boundaries = check_boundaries
        #############VI################
        #############VI################
        # Without Time
        self.my_initial = self.convert_to_my_state(initial)
        self.states = []
        self.generate_states(initial)
        self.values = {}  # (state, turn): value
        self.policy = {}  # (state, turn): action

        for state in self.states:
            self.values[(state, 0)] = 0
            self.policy[(state, 0)] = None # No action at the very end

        self.compute_values_and_policy()

        self.value_iteration_with_policy()

    def convert_to_my_state(self, state):

        # Pirates
        pirates = [(pirate_key, pirate.get("location"), pirate.get("capacity")) for pirate_key, pirate in
                   state.get("pirate_ships").items()]

        # Treasures
        treasures = [(treasure_key, treasure.get("location")) for treasure_key, treasure in
                     state.get("treasures").items()]

        marines = [(marine_key, marine.get("index")) for marine_key, marine in state.get("marine_ships").items()]

        # steps_left = int(state.get("turns to go"))

        my_representation = (tuple(pirates),) + (tuple(treasures),) + (tuple(marines),)

        return my_representation

    def actions(self, state):

        pir_names = [state[0][j][0] for j in range(len(state[0]))]
        pir_locations = [state[0][j][1] for j in range(len(state[0]))]
        pirates = state[0]
        treasures = state[1]
        marines = state[2]
        totalActions = ()

        for i, (name, location) in enumerate(zip(pir_names, pir_locations)):
            found = 0
            actions = ()

            # Sail
            if (location[0] < self.gridRowsNum - 1) and (self.grid[location[0] + 1][location[1]] != 'I'):  # down
                if (location[0] + 1, location[1]):
                    action_move = ("sail",) + (name,) + ((location[0] + 1, location[1]),)
                    actions = actions + (action_move,)
            if (location[0] > 0) and (self.grid[location[0] - 1][location[1]] != 'I'):  # up
                if (location[0] - 1, location[1]):
                    action_move = ("sail",) + (name,) + ((location[0] - 1, location[1]),)
                    actions = actions + (action_move,)
            if (location[1] < self.gridColumnsNum - 1) and (self.grid[location[0]][location[1] + 1] != 'I'):  # right
                if (location[0], location[1] + 1):
                    action_move = ("sail",) + (name,) + ((location[0], location[1] + 1),)
                    actions = actions + (action_move,)
            if (location[1] > 0) and (self.grid[location[0]][location[1] - 1] != 'I'):  # left
                if (location[0], location[1] - 1):
                    action_move = ("sail",) + (name,) + ((location[0], location[1] - 1),)
                    actions = actions + (action_move,)

            # dock and deposit a treasure
            if (self.grid[location[0]][location[1]] == 'B') and pirates[i][2] < 2:
                action_deposit = ("deposit",) + (name,)
                actions = actions + (action_deposit,)
                found = 1

            # Collect
            for treasure in treasures:
                # DOWN
                if self.check_boundaries[location[0] + 2][location[1] + 1] != 'W':
                    if (self.grid[location[0] + 1][location[1]]) == 'I':
                        if treasure[1] == (location[0] + 1, location[1]) and pirates[i][2] > 0:
                            action_move = ("collect",) + (name,) + (treasure[0],)
                            actions = actions + (action_move,)
                # UP
                if self.check_boundaries[location[0]][location[1] + 1] != 'W':
                    if (self.grid[location[0] - 1][location[1]]) == 'I':
                        if treasure[1] == (location[0] - 1, location[1]) and pirates[i][2] > 0:
                            action_move = ("collect",) + (name,) + (treasure[0],)
                            actions = actions + (action_move,)
                # RIGHT
                if self.check_boundaries[location[0] + 1][location[1] + 2] != 'W':
                    if (self.grid[location[0]][location[1] + 1]) == 'I':
                        if treasure[1] == (location[0], location[1] + 1) and pirates[i][2] > 0:
                            action_move = ("collect",) + (name,) + (treasure[0],)
                            actions = actions + (action_move,)
                # LEFT
                if self.check_boundaries[location[0] + 1][location[1]] != 'W':
                    if (self.grid[location[0]][location[1] - 1]) == 'I':
                        if treasure[1] == (location[0], location[1] - 1) and pirates[i][2] > 0:
                            action_move = ("collect",) + (name,) + (treasure[0],)
                            actions = actions + (action_move,)

            if found == 0:
                # Wait
                action_wait = ("wait",) + (name,)
                actions = actions + (action_wait,)

            # all actions for all users
            totalActions = totalActions + (actions,)

        # atomic actions
        final_atomic_actions = list(itertools.product(*totalActions))

        final_atomic_actions.append("reset")
        # final_atomic_actions.append("terminate")

        return tuple(final_atomic_actions)

    def resultWithoutTime(self, state, action):
        if action == "reset": return ((self.my_initial),)
        # if action == "terminate": return ((self.initialWithoutTime),)

        # Opening the tuples
        ############################
        pirates = list(state[0])
        treasures = list(state[1])
        marines = list(state[2])
        ############################

        # Marines Combinations
        marine_change = []
        marines_changes = []
        for i, marine in enumerate(marines):
            # Path size = 1
            if (len(self.marines_info[i][1]) == 1):
                marine_change.append([(marine[0], marine[1])])
                continue
            # Left side
            elif (marine[1] == 0):
                marine_change.append([(marine[0], 0), (marine[0], 1)])
            # Right side
            elif (marine[1] == len(self.marines_info[i][1]) - 1):
                marine_change.append([(marine[0], marine[1] - 1), (marine[0], marine[1])])
            # Middle
            else:
                marine_change.append([(marine[0], marine[1] - 1), (marine[0], marine[1]), (marine[0], marine[1] + 1)])
        # Create combinations
        marines_changes = list(itertools.product(*marine_change))

        # Treasures Combinations
        treasure_change = []
        treasures_changes = []
        for j, treasure in enumerate(treasures):
            # Probability zero : stay still
            if self.treasures_info[j][2] == 0:
                treasure_change.append([(treasure[0], treasure[1])])
            else:
                treasure_change.append(
                    [(treasure[0], possible_location) for possible_location in self.treasures_info[j][1]])
                treasure_change[j].append((treasure[0], treasure[1]))
                treasure_change[j] = list(set(treasure_change[j]))
        # Create combinations
        treasures_changes = list(itertools.product(*treasure_change))

        # Deterministic Pirates
        catches = []

        for i, act in enumerate(action):
            # Pirate point of view
            if act[0] == "sail":
                pirates[i] = (pirates[i][0], act[2], pirates[i][2])
            if act[0] == "deposit":
                pirates[i] = (pirates[i][0], pirates[i][1], 2)
            if act[0] == "collect":
                pirates[i] = (pirates[i][0], pirates[i][1], pirates[i][2] - 1)
            if act[0] == "wait":
                pirates[i] = (pirates[i][0], pirates[i][1], pirates[i][2])

            catchesByOption = 0
            for marines_option in marines_changes:
                for k, marine in enumerate(marines_option):
                    if self.marines_info[k][1][marine[1]] == pirates[i][1]:
                        catchesByOption += 1
                        break
            catches.append(catchesByOption)

        # Pirates Capacity Combinations
        pirate_change = []
        pirates_changes = []

        for i, pirate in enumerate(pirates):
            if catches[i] == len(marines_changes):
                pirate_change.append([(pirate[0], pirate[1], 2)])
                continue
            pirate_change.append([(pirates[i])])
        pirates_changes = list(itertools.product(*pirate_change))
        pirates_changes = list(set(pirates_changes))

        # All combinations
        parts = (tuple(pirates_changes),) + (tuple(treasures_changes),) + (tuple(marines_changes),)
        all_resulted_states = list(itertools.product(*parts))

        nested_list = [[list(inner_tuple) for inner_tuple in list(outer_tuple)] for outer_tuple in all_resulted_states]

        for j, option in enumerate(nested_list):
            option = list(option)
            for i, pirate in enumerate(option[0]):
                for k, marine in enumerate(option[2]):
                    if self.marines_info[k][1][marine[1]] == option[0][i][1]:
                        if option[0][i][2] != 2:
                            # Change the value of option[0][i][2] to 2
                            nested_list[j][0][i] = (option[0][i][0], option[0][i][1], 2)
                            # Append the modified option to new_tuple

        nested_tuples = tuple(
            tuple(tuple(inner_tuple) for inner_tuple in outer_tuple) for outer_tuple in all_resulted_states)

        return tuple(nested_tuples)

    def generate_states(self, initial):
        init_state = self.convert_to_my_state(initial)
        pirates = init_state[0]
        treasures = init_state[1]
        marines = init_state[2]

        pirate_possible_locations = []
        for i, row in enumerate(self.grid):
            for j in range(len(row)):
                if self.grid[i][j] != "I": pirate_possible_locations.append((i, j))

        pirates_possibilities = ()
        for pirate in pirates:
            pirate_poss = ()
            for poss in pirate_possible_locations:
                for i in range(3):
                    pirate_poss = pirate_poss + ((pirate[0], poss, i),)
            pirates_possibilities = pirates_possibilities + (pirate_poss,)

        treasures_possibilites = ()
        for i, treasure in enumerate(treasures):
            tres_poss = ()
            tres_poss = tres_poss + ((treasure[0], treasure[1]),)
            for loc in self.treasures_info[i][1]:
                tres_poss = tres_poss + ((treasure[0], loc),)
            tres_poss = set(tres_poss)
            treasures_possibilites = treasures_possibilites + (tres_poss,)

        marines_possibilites = ()
        for i, marine in enumerate(marines):
            marine_poss = ()
            for j in range(len(self.marines_info[i][1])):
                marine_poss = marine_poss + ((marine[0], j),)
            marines_possibilites = marines_possibilites + (marine_poss,)

        pirates_combinations = list(itertools.product(*pirates_possibilities))
        treasures_combinations = list(itertools.product(*treasures_possibilites))
        marines_combinations = list(itertools.product(*marines_possibilites))

        combined_lists = []
        combined_lists.append(pirates_combinations)
        combined_lists.append(treasures_combinations)
        combined_lists.append(marines_combinations)


        self.states = list(itertools.product(*combined_lists))

    def transition_probabilities_and_rewards(self, state, action):
        transitions_and_rewards = []

        # Use resultWithoutTime to get all possible next states
        possible_next_states = self.resultWithoutTime(state, action)

        # For each possible next state, calculate the transition probability and reward
        for next_state in possible_next_states:
            prob = self.transitions(state, action, next_state)
            reward = self.get_R(state, action, next_state)
            transitions_and_rewards.append((next_state, prob, reward))

        return transitions_and_rewards

    def compute_value_and_best_action(self, state, turn):

        max_value = float('-inf')
        best_action = None
        for action in self.actions(state):
            total_value = 0
            transitions = self.transition_probabilities_and_rewards(state, action)
            for next_state, prob, reward in transitions:
                future_value = self.values.get((next_state, turn - 1), 0)
                total_value += prob * (reward + future_value)
            if total_value > max_value:
                max_value = total_value
                best_action = action

        return max_value, best_action

    def compute_values_and_policy(self):
        # For each turn from the last to the first, calculate values and policy
        for turn in range(1, self.horizon + 1):
            for state in self.states:
                self.values[(state, turn)], self.policy[(state, turn)] = self.compute_value_and_best_action(state, turn)

    def get_R(self, state, action, next_state):

        pirates_old = state[0]
        pirates_new = next_state[0]
        marines = next_state[2]

        total_reward = 0

        # Case 1
        if action == "reset":
            return -2


        for i, act in enumerate(action):

            # Case 2
            if act[0] == "deposit":
                total_reward += (2 - pirates_old[i][2]) * 4
            # Case 3
            for k, marine in enumerate(marines):
                if self.marines_info[k][1][marine[1]] == pirates_new[i][1]:
                    total_reward -= 1
                    break
        return total_reward

    def transitions(self, state, action, next_state):

        if action == 'reset':
            return 1

        pirates = state[0]
        treasures = state[1]
        treasures_next_state = next_state[1]
        marines = state[2]
        marines_next_state = next_state[2]

        # Marine Movement
        marines_prob = {}
        for i, marine in enumerate(marines):
            # Only one location
            if len(self.marines_info[i][1]) == 1:
                marines_prob[marine[0]] = 1

            # On sides of the path

            elif ((marine[1] == 0) or (marine[1] == len(self.marines_info[i][1]) - 1)):
                marines_prob[marine[0]] = 1 / 2

            else:
                marines_prob[marine[0]] = (1 / 3)

        # Treasure movement
        treasures_prob = {}
        for i, treasure in enumerate(treasures):
            change_prob = self.treasures_info[i][2]
            num_of_locations = len(self.treasures_info[i][1])
            move_prob = (1 / num_of_locations) * change_prob
            my_location_count = 0

            for location in self.treasures_info[i][1]:
                if treasure[1] == location:
                    my_location_count += 1

            # Stay on same place
            if treasure[1] == treasures_next_state[i][1]:
                treasures_prob[treasure[0]] = (1 - change_prob) + my_location_count * move_prob
            else:
                treasures_prob[treasure[0]] = move_prob

        total_marine_prob = 1
        for marine in marines_prob:
            total_marine_prob *= marines_prob[marine]

        total_treasure_prob = 1
        for treasure in treasures_prob:
            total_treasure_prob *= treasures_prob[treasure]

        return total_marine_prob * total_treasure_prob

    def value_iteration_with_policy(self):
        # Initialize value function and policy. Each state already includes time, so we don't pair it with time again.
        V = dict.fromkeys(self.states, 0)  # Value function
        # updated_V = {state: 0 for state in self.states}
        keys = [(s, t) for s in self.states for t in range(self.horizon)]
        best_actions = dict.fromkeys(keys, 0)
        all_value_functions = []

        # policy = {state: None for state in self.states}  # Policy
        for t in range(self.horizon):

            updated_V = dict.fromkeys(self.states, 0)
            # V = updated_V.copy()
            for state in self.states:
                max_value = float('-inf')
                arg_best_action = 0

                for action in self.actions(state):
                    total_prob = 0
                    expected_value = 0
                    possible_states = self.resultWithoutTime(state, action)
                    for next_state in possible_states:
                        probability = self.transitions(state, action, next_state)
                        total_prob += probability
                        # Here, we ensure that the reward is added correctly
                        expected_value += probability * (V[next_state])
                        expected_value += self.get_R(state, action, next_state)

                    if expected_value > max_value:
                        max_value = expected_value
                        arg_best_action = action

                updated_V[state] = max_value
                best_actions[(state, t)] = arg_best_action
            # Update the value function for the next iteration

            V = updated_V.copy()

            all_value_functions.append(updated_V)

        self.policy = best_actions
        return all_value_functions

    def act(self, state):
        time = int(state.get("turns to go"))
        new_state = deepcopy(state)
        new_state["turns to go"] = self.horizon
        current_turn = time - 1
        new_state = self.convert_to_my_state(new_state)
        best_act = self.policy[(new_state, current_turn)]
        if best_act == None :
            best_act = 'terminate'
        if current_turn == 0:
            return best_act

        return best_act


class PirateAgent:
    def __init__(self, initial):

        self.start = time.time()

        self.grid = tuple(tuple(sub) for sub in initial.get("map"))
        self.gridRowsNum = len(self.grid)
        self.gridColumnsNum = len(self.grid[0])
        self.treasures_info = [(treasure_key, treasure.get("possible_locations"), treasure.get("prob_change_location"))
                               for treasure_key, treasure in initial.get("treasures").items()]
        self.marines_info = [(marine_key, marine.get("path")) for marine_key, marine in
                             initial.get("marine_ships").items()]
        self.horizon = int(initial.get("turns to go"))

        original_map = initial["map"]
        rows = len(original_map)
        cols = len(original_map[0]) if rows > 0 else 0
        expanded_rows = rows + 2
        expanded_cols = cols + 2
        check_boundaries = [['W' for _ in range(expanded_cols)] for _ in range(expanded_rows)]
        for i in range(rows):
            for j in range(cols):
                check_boundaries[i + 1][j + 1] = original_map[i][j]
        self.check_boundaries = check_boundaries

        ###############################################################################

        self.states = []

        # Final representation
        my_initial = self.relax(initial)

        self.initialWithoutTime = my_initial
        self.states = self.generate_states(my_initial)
        # print(len(self.states))
        # self.bfs(my_initial, self.horizon)
        self.values = {}  # (state, turn): value
        self.policy = {}  # (state, turn): action

        for state in self.states:
            self.values[(state, 0)] = 0
            self.policy[(state, 0)] = None  # No action at the very end

        self.compute_values_and_policy()



    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def evaluate_treasure_advanced(self, state):

        treasures = state["treasures"]
        best_score = float('-inf')
        best_treasure = None
        base_loc = (0, 0)


        for treasure in treasures.items():
            # Calculate average distance to all pirates
            avg_distance = sum([self.manhattan_distance(treasure["location"], base_loc) ]) /len(set(treasure["possible_locations"])))

            # Stability Score (inversely proportional to the number of possible locations)
            stability_score = 1 / len(set(info["possible_locations"]))

            # Probability of Change
            prob_change_score = 1 - info["prob_change_location"]

            # Length of Possible Locations Array
            length_score = 1 / len(info["possible_locations"])


            # Combined score
            score = stability_score + prob_change_score + (
                        1 / (avg_distance + 1)) + length_score

            if score > best_score:
                best_score = score
                best_treasure = treasure

        return best_treasure




    def relax(self, state):

        # Pirates
        pirates = [(pirate_key, pirate.get("location"), pirate.get("capacity")) for pirate_key, pirate in
                   state.get("pirate_ships").items()]

        # Treasures
        best_treasure = self.evaluate_treasure_advanced(state)
        print(best_treasure)
        treasures = [(treasure_key, treasure.get("location")) for treasure_key, treasure in
                     state.get("treasures").items() if treasure_key == best_treasure]

        marines = [(marine_key, marine.get("index")) for marine_key, marine in state.get("marine_ships").items()]

        my_representation = ((tuple(pirates[0]),),) + ((tuple(treasures[0]),),) + ((tuple(marines[0]),),)

        return my_representation

    def convert_to_my_state(self, state):

        # Pirates
        pirates = [(pirate_key, pirate.get("location"), pirate.get("capacity")) for pirate_key, pirate in
                   state.get("pirate_ships").items()]

        # Treasures
        treasures = [(treasure_key, treasure.get("location")) for treasure_key, treasure in
                     state.get("treasures").items()]

        marines = [(marine_key, marine.get("index")) for marine_key, marine in state.get("marine_ships").items()]

        my_representation = (tuple(pirates),) + (tuple(treasures),) + (tuple(marines),)

        return my_representation

    def actions(self, state):

        pir_names = [state[0][j][0] for j in range(len(state[0]))]
        pir_locations = [state[0][j][1] for j in range(len(state[0]))]
        pirates = state[0]
        treasures = state[1]
        marines = state[2]
        totalActions = ()

        for i, (name, location) in enumerate(zip(pir_names, pir_locations)):
            found = 0
            actions = ()

            # Sail
            if (location[0] < self.gridRowsNum - 1) and (self.grid[location[0] + 1][location[1]] != 'I'):  # down
                if (location[0] + 1, location[1]):
                    action_move = ("sail",) + (name,) + ((location[0] + 1, location[1]),)
                    actions = actions + (action_move,)
            if (location[0] > 0) and (self.grid[location[0] - 1][location[1]] != 'I'):  # up
                if (location[0] - 1, location[1]):
                    action_move = ("sail",) + (name,) + ((location[0] - 1, location[1]),)
                    actions = actions + (action_move,)
            if (location[1] < self.gridColumnsNum - 1) and (self.grid[location[0]][location[1] + 1] != 'I'):  # right
                if (location[0], location[1] + 1):
                    action_move = ("sail",) + (name,) + ((location[0], location[1] + 1),)
                    actions = actions + (action_move,)
            if (location[1] > 0) and (self.grid[location[0]][location[1] - 1] != 'I'):  # left
                if (location[0], location[1] - 1):
                    action_move = ("sail",) + (name,) + ((location[0], location[1] - 1),)
                    actions = actions + (action_move,)

            # dock and deposit a treasure
            if (self.grid[location[0]][location[1]] == 'B') and pirates[i][2] < 2:
                action_deposit = ("deposit",) + (name,)
                actions = actions + (action_deposit,)
                found = 1

            # Collect
            for treasure in treasures:
                # DOWN
                if self.check_boundaries[location[0] + 2][location[1] + 1] != 'W':
                    if (self.grid[location[0] + 1][location[1]]) == 'I':
                        if treasure[1] == (location[0] + 1, location[1]) and pirates[i][2] > 0:
                            action_move = ("collect",) + (name,) + (treasure[0],)
                            actions = actions + (action_move,)
                # UP
                if self.check_boundaries[location[0]][location[1] + 1] != 'W':
                    if (self.grid[location[0] - 1][location[1]]) == 'I':
                        if treasure[1] == (location[0] - 1, location[1]) and pirates[i][2] > 0:
                            action_move = ("collect",) + (name,) + (treasure[0],)
                            actions = actions + (action_move,)
                # RIGHT
                if self.check_boundaries[location[0] + 1][location[1] + 2] != 'W':
                    if (self.grid[location[0]][location[1] + 1]) == 'I':
                        if treasure[1] == (location[0], location[1] + 1) and pirates[i][2] > 0:
                            action_move = ("collect",) + (name,) + (treasure[0],)
                            actions = actions + (action_move,)
                # LEFT
                if self.check_boundaries[location[0] + 1][location[1]] != 'W':
                    if (self.grid[location[0]][location[1] - 1]) == 'I':
                        if treasure[1] == (location[0], location[1] - 1) and pirates[i][2] > 0:
                            action_move = ("collect",) + (name,) + (treasure[0],)
                            actions = actions + (action_move,)

                # Wait
                action_wait = ("wait",) + (name,)
                actions = actions + (action_wait,)

            # all actions for all users
            totalActions = totalActions + (actions,)

        # atomic actions
        final_atomic_actions = list(itertools.product(*totalActions))

        final_atomic_actions.append("reset")
        # final_atomic_actions.append("terminate")

        return tuple(final_atomic_actions)

    def resultWithoutTime(self, state, action):
        if action == "reset": return ((self.initialWithoutTime),)
        # if action == "terminate": return ((self.initialWithoutTime),)

        # Opening the tuples
        ############################
        pirates = list(state[0])
        treasures = list(state[1])
        marines = list(state[2])
        ############################

        # Marines Combinations
        marine_change = []
        marines_changes = []
        for i, marine in enumerate(marines):
            # Path size = 1
            if (len(self.marines_info[i][1]) == 1):
                marine_change.append([(marine[0], marine[1])])
                continue
            # Left side
            elif (marine[1] == 0):
                marine_change.append([(marine[0], 0), (marine[0], 1)])
            # Right side
            elif (marine[1] == len(self.marines_info[i][1]) - 1):
                marine_change.append([(marine[0], marine[1] - 1), (marine[0], marine[1])])
            # Middle
            else:
                marine_change.append([(marine[0], marine[1] - 1), (marine[0], marine[1]), (marine[0], marine[1] + 1)])
        # Create combinations
        marines_changes = list(itertools.product(*marine_change))

        # Treasures Combinations
        treasure_change = []
        treasures_changes = []
        for j, treasure in enumerate(treasures):
            # Probability zero : stay still
            if self.treasures_info[j][2] == 0:
                treasure_change.append([(treasure[0], treasure[1])])
            else:
                treasure_change.append(
                    [(treasure[0], possible_location) for possible_location in self.treasures_info[j][1]])
                treasure_change[j].append((treasure[0], treasure[1]))
                treasure_change[j] = list(set(treasure_change[j]))
        # Create combinations
        treasures_changes = list(itertools.product(*treasure_change))

        # Deterministic Pirates
        catches = []

        for i, act in enumerate(action):
            # Pirate point of view
            if act[0] == "sail":
                pirates[i] = (pirates[i][0], act[2], pirates[i][2])
            if act[0] == "deposit":
                pirates[i] = (pirates[i][0], pirates[i][1], 2)
            if act[0] == "collect":
                pirates[i] = (pirates[i][0], pirates[i][1], pirates[i][2] - 1)
            if act[0] == "wait":
                pirates[i] = (pirates[i][0], pirates[i][1], pirates[i][2])

            catchesByOption = 0
            for marines_option in marines_changes:
                for k, marine in enumerate(marines_option):
                    if self.marines_info[k][1][marine[1]] == pirates[i][1]:
                        catchesByOption += 1
                        break
            catches.append(catchesByOption)

        # Pirates Capacity Combinations
        pirate_change = []
        pirates_changes = []

        for i, pirate in enumerate(pirates):
            if catches[i] == len(marines_changes):
                pirate_change.append([(pirate[0], pirate[1], 2)])
                continue
            pirate_change.append([(pirates[i])])
        pirates_changes = list(itertools.product(*pirate_change))
        pirates_changes = list(set(pirates_changes))

        # All combinations
        parts = (tuple(pirates_changes),) + (tuple(treasures_changes),) + (tuple(marines_changes),)
        all_resulted_states = list(itertools.product(*parts))

        nested_list = [[list(inner_tuple) for inner_tuple in list(outer_tuple)] for outer_tuple in all_resulted_states]

        for j, option in enumerate(nested_list):
            option = list(option)
            for i, pirate in enumerate(option[0]):
                for k, marine in enumerate(option[2]):
                    if self.marines_info[k][1][marine[1]] == option[0][i][1]:
                        if option[0][i][2] != 2:
                            # Change the value of option[0][i][2] to 2
                            nested_list[j][0][i] = (option[0][i][0], option[0][i][1], 2)
                            # Append the modified option to new_tuple

        nested_tuples = tuple(
            tuple(tuple(inner_tuple) for inner_tuple in outer_tuple) for outer_tuple in all_resulted_states)

        return tuple(nested_tuples)

    def bfs(self, initial_state, max_depth):
        queue = deque([(initial_state, max_depth)])
        while queue:
            state, depth = queue.popleft()
            if depth < 0:
                return
            if state not in self.states:
                self.states.append(state)
                for action in self.actions(state):
                    next_states = list(self.resultWithoutTime(state, action))  # CHANGE
                    for next_state in next_states:
                        queue.append((next_state, depth - 1))

    def generate_states(self, initial):
        # init_state = self.relax(initial)
        pirates = initial[0]
        treasures = initial[1]
        marines = initial[2]

        pirate_possible_locations = []
        for i, row in enumerate(self.grid):
            for j in range(len(row)):
                if self.grid[i][j] != "I": pirate_possible_locations.append((i, j))

        pirates_possibilities = ()
        for pirate in pirates:
            pirate_poss = ()
            for poss in pirate_possible_locations:
                for i in range(3):
                    pirate_poss = pirate_poss + ((pirate[0], poss, i),)
            pirates_possibilities = pirates_possibilities + (pirate_poss,)

        treasures_possibilites = ()
        for i, treasure in enumerate(treasures):
            tres_poss = ()
            tres_poss = tres_poss + ((treasure[0], treasure[1]),)
            for loc in self.treasures_info[i][1]:
                tres_poss = tres_poss + ((treasure[0], loc),)
            tres_poss = set(tres_poss)
            treasures_possibilites = treasures_possibilites + (tres_poss,)

        marines_possibilites = ()
        for i, marine in enumerate(marines):
            marine_poss = ()
            for j in range(len(self.marines_info[i][1])):
                marine_poss = marine_poss + ((marine[0], j),)
            marines_possibilites = marines_possibilites + (marine_poss,)


        pirates_combinations = list(itertools.product(*pirates_possibilities))
        treasures_combinations = list(itertools.product(*treasures_possibilites))
        marines_combinations = list(itertools.product(*marines_possibilites))

        combined_lists = []
        combined_lists.append(pirates_combinations)
        combined_lists.append(treasures_combinations)
        combined_lists.append(marines_combinations)

        all_combinations = list(itertools.product(*combined_lists))

        return all_combinations

    def transition_probabilities_and_rewards(self, state, action):
        transitions_and_rewards = []

        # Use resultWithoutTime to get all possible next states
        possible_next_states = self.resultWithoutTime(state, action)

        # For each possible next state, calculate the transition probability and reward
        for next_state in possible_next_states:
            prob = self.transitions(state, action, next_state)
            reward = self.get_R(state, action, next_state)
            transitions_and_rewards.append((next_state, prob, reward))

        return transitions_and_rewards

    def compute_value_and_best_action(self, state, turn):

        max_value = float('-inf')
        best_action = None
        for action in self.actions(state):
            total_value = 0
            transitions = self.transition_probabilities_and_rewards(state, action)
            for next_state, prob, reward in transitions:
                future_value = self.values.get((next_state, turn - 1), 0)
                total_value += prob * (reward + future_value)
            if total_value > max_value:
                max_value = total_value
                best_action = action

        return max_value, best_action

    def compute_values_and_policy(self):
        # For each turn from the last to the first, calculate values and policy
        for turn in range(1, self.horizon + 1):
            for state in self.states:
                self.values[(state, turn)], self.policy[(state, turn)] = self.compute_value_and_best_action(state, turn)

    def get_R(self, state, action, next_state):

        pirates_old = state[0]
        pirates_new = next_state[0]
        marines = next_state[2]

        total_reward = 0

        # Case 1
        if action == "reset":
            return -2

        for i, act in enumerate(action):

            # Case 2
            if act[0] == "deposit":
                total_reward += (2 - pirates_old[i][2]) * 4
            # Case 3
            for k, marine in enumerate(marines):
                if self.marines_info[k][1][marine[1]] == pirates_new[i][1]:
                    total_reward -= 1
                    break
        return total_reward

    def transitions(self, state, action, next_state):

        if action == 'reset':
            return 1

        pirates = state[0]
        treasures = state[1]
        treasures_next_state = next_state[1]
        marines = state[2]
        marines_next_state = next_state[2]

        # Marine Movement
        marines_prob = {}
        for i, marine in enumerate(marines):
            # Only one location
            if len(self.marines_info[i][1]) == 1:
                marines_prob[marine[0]] = 1

            # On sides of the path

            elif ((marine[1] == 0) or (marine[1] == len(self.marines_info[i][1]) - 1)):
                marines_prob[marine[0]] = 1 / 2

            else:
                marines_prob[marine[0]] = (1 / 3)

        # Treasure movement
        treasures_prob = {}
        for i, treasure in enumerate(treasures):
            change_prob = self.treasures_info[i][2]
            num_of_locations = len(self.treasures_info[i][1])
            move_prob = (1 / num_of_locations) * change_prob
            my_location_count = 0

            for location in self.treasures_info[i][1]:
                if treasure[1] == location:
                    my_location_count += 1

            # Stay on same place
            if treasure[1] == treasures_next_state[i][1]:
                treasures_prob[treasure[0]] = (1 - change_prob) + my_location_count * move_prob
            else:
                treasures_prob[treasure[0]] = move_prob

        total_marine_prob = 1
        for marine in marines_prob:
            total_marine_prob *= marines_prob[marine]

        total_treasure_prob = 1
        for treasure in treasures_prob:
            total_treasure_prob *= treasures_prob[treasure]

        return total_marine_prob * total_treasure_prob


    def act(self, state):
        t = state["turns to go"]
        new_state = deepcopy(state)
        new_state["turns to go"] = self.horizon
        current_turn = t - 1
        relaxed_state = self.relax(new_state)
        action = self.policy[(relaxed_state, current_turn)]
        if action == None:
            action = 'terminate'

        if action == "reset" or action == "terminate":
            return action

        # Same action for each pirate
        actions_multiplied = ()
        for i in range(len(state["pirate_ships"])):
            actions_multiplied = actions_multiplied + (action[0] ,)
        list_of_lists = [list(inner_tuple) for inner_tuple in actions_multiplied]

        # Changing the name of the pirates for each action
        for i, pirate_name in enumerate(state["pirate_ships"]):
            list_of_lists[i][0] = action[0][0]
            list_of_lists[i][1] = pirate_name

        final_duplicated_action = tuple(map(tuple, list_of_lists))

        return final_duplicated_action


class InfinitePirateAgent:
    def __init__(self, initial, gamma):

        # Gamma
        self.gamma = gamma


        self.grid = tuple(tuple(sub) for sub in initial.get("map"))
        self.gridRowsNum = len(self.grid)
        self.gridColumnsNum = len(self.grid[0])
        self.treasures_info = [(treasure_key, treasure.get("possible_locations"), treasure.get("prob_change_location"))
                               for treasure_key, treasure in initial.get("treasures").items()]
        self.marines_info = [(marine_key, marine.get("path")) for marine_key, marine in
                             initial.get("marine_ships").items()]

        #self.horizon = int(initial.get("turns to go"))

        original_map = initial["map"]
        rows = len(original_map)
        cols = len(original_map[0]) if rows > 0 else 0
        expanded_rows = rows + 2
        expanded_cols = cols + 2
        check_boundaries = [['W' for _ in range(expanded_cols)] for _ in range(expanded_rows)]
        for i in range(rows):
            for j in range(cols):
                check_boundaries[i + 1][j + 1] = original_map[i][j]
        self.check_boundaries = check_boundaries


        ###############################################################################

        self.states = []

        # Final representation
        my_initial = self.convert_to_my_state(initial)
        self.initialWithoutTime = my_initial
        self.states = self.generate_states(initial)
        #self.bfs(my_initial, self.horizon)
        self.policy = {}

        self.value_iteration()

    def convert_to_my_state(self, state):

        # Pirates
        pirates = [(pirate_key, pirate.get("location"), pirate.get("capacity")) for pirate_key, pirate in
                   state.get("pirate_ships").items()]

        # Treasures
        treasures = [(treasure_key, treasure.get("location")) for treasure_key, treasure in
                     state.get("treasures").items()]

        marines = [(marine_key, marine.get("index")) for marine_key, marine in state.get("marine_ships").items()]

        my_representation = (tuple(pirates),) + (tuple(treasures),) + (tuple(marines),)

        return my_representation

    def actions(self, state):

        pir_names = [state[0][j][0] for j in range(len(state[0]))]
        pir_locations = [state[0][j][1] for j in range(len(state[0]))]
        pirates = state[0]
        treasures = state[1]
        marines = state[2]
        totalActions = ()

        for i, (name, location) in enumerate(zip(pir_names, pir_locations)):
            found = 0
            actions = ()

            # Sail
            if (location[0] < self.gridRowsNum - 1) and (self.grid[location[0] + 1][location[1]] != 'I'):  # down
                if (location[0] + 1, location[1]):
                    action_move = ("sail",) + (name,) + ((location[0] + 1, location[1]),)
                    actions = actions + (action_move,)
            if (location[0] > 0) and (self.grid[location[0] - 1][location[1]] != 'I'):  # up
                if (location[0] - 1, location[1]):
                    action_move = ("sail",) + (name,) + ((location[0] - 1, location[1]),)
                    actions = actions + (action_move,)
            if (location[1] < self.gridColumnsNum - 1) and (self.grid[location[0]][location[1] + 1] != 'I'):  # right
                if (location[0], location[1] + 1):
                    action_move = ("sail",) + (name,) + ((location[0], location[1] + 1),)
                    actions = actions + (action_move,)
            if (location[1] > 0) and (self.grid[location[0]][location[1] - 1] != 'I'):  # left
                if (location[0], location[1] - 1):
                    action_move = ("sail",) + (name,) + ((location[0], location[1] - 1),)
                    actions = actions + (action_move,)

            # dock and deposit a treasure
            if (self.grid[location[0]][location[1]] == 'B') and pirates[i][2] < 2:
                action_deposit = ("deposit",) + (name,)
                actions = actions + (action_deposit,)
                found = 1

            # Collect
            for treasure in treasures:
                # DOWN
                if self.check_boundaries[location[0] + 2][location[1] + 1] != 'W':
                    if (self.grid[location[0] + 1][location[1]]) == 'I':
                        if treasure[1] == (location[0] + 1, location[1]) and pirates[i][2] > 0:
                            action_move = ("collect",) + (name,) + (treasure[0],)
                            actions = actions + (action_move,)
                # UP
                if self.check_boundaries[location[0]][location[1] + 1] != 'W':
                    if (self.grid[location[0] - 1][location[1]]) == 'I':
                        if treasure[1] == (location[0] - 1, location[1]) and pirates[i][2] > 0:
                            action_move = ("collect",) + (name,) + (treasure[0],)
                            actions = actions + (action_move,)
                # RIGHT
                if self.check_boundaries[location[0] + 1][location[1] + 2] != 'W':
                    if (self.grid[location[0]][location[1] + 1]) == 'I':
                        if treasure[1] == (location[0], location[1] + 1) and pirates[i][2] > 0:
                            action_move = ("collect",) + (name,) + (treasure[0],)
                            actions = actions + (action_move,)
                # LEFT
                if self.check_boundaries[location[0] + 1][location[1]] != 'W':
                    if (self.grid[location[0]][location[1] - 1]) == 'I':
                        if treasure[1] == (location[0], location[1] - 1) and pirates[i][2] > 0:
                            action_move = ("collect",) + (name,) + (treasure[0],)
                            actions = actions + (action_move,)


                # Wait
                action_wait = ("wait",) + (name,)
                actions = actions + (action_wait,)

            # all actions for all users
            totalActions = totalActions + (actions,)

        # atomic actions
        final_atomic_actions = list(itertools.product(*totalActions))

        final_atomic_actions.append("reset")
        # final_atomic_actions.append("terminate")

        return tuple(final_atomic_actions)

    def resultWithoutTime(self, state, action):
        if action == "reset": return ((self.initialWithoutTime),)
        # if action == "terminate": return ((self.initialWithoutTime),)

        # Opening the tuples
        ############################
        pirates = list(state[0])
        treasures = list(state[1])
        marines = list(state[2])
        ############################

        # Marines Combinations
        marine_change = []
        marines_changes = []
        for i, marine in enumerate(marines):
            # Path size = 1
            if (len(self.marines_info[i][1]) == 1):
                marine_change.append([(marine[0], marine[1])])
                continue
            # Left side
            elif (marine[1] == 0):
                marine_change.append([(marine[0], 0) , (marine[0], 1)])
            # Right side
            elif (marine[1] == len(self.marines_info[i][1]) - 1):
                marine_change.append([(marine[0], marine[1] - 1) , (marine[0], marine[1])])
            # Middle
            else:
                marine_change.append([(marine[0], marine[1] - 1), (marine[0], marine[1]), (marine[0], marine[1] + 1)])
        # Create combinations
        marines_changes = list(itertools.product(*marine_change))

        # Treasures Combinations
        treasure_change = []
        treasures_changes = []
        for j, treasure in enumerate(treasures):
            # Probability zero : stay still
            if self.treasures_info[j][2] == 0:
                treasure_change.append([(treasure[0], treasure[1])])
            else:
                treasure_change.append(
                    [(treasure[0], possible_location) for possible_location in self.treasures_info[j][1]])
                treasure_change[j].append((treasure[0], treasure[1]))
                treasure_change[j] = list(set(treasure_change[j]))
        # Create combinations
        treasures_changes = list(itertools.product(*treasure_change))

        # Deterministic Pirates
        catches = []

        for i, act in enumerate(action):
            # Pirate point of view
            if act[0] == "sail":
                pirates[i] = (pirates[i][0], act[2], pirates[i][2])
            if act[0] == "deposit":
                pirates[i] = (pirates[i][0], pirates[i][1], 2)
            if act[0] == "collect":
                pirates[i] = (pirates[i][0], pirates[i][1], pirates[i][2] - 1)
            if act[0] == "wait":
                pirates[i] = (pirates[i][0], pirates[i][1], pirates[i][2])

            catchesByOption = 0
            for marines_option in marines_changes:
                for k, marine in enumerate(marines_option):
                    if self.marines_info[k][1][marine[1]] == pirates[i][1]:
                        catchesByOption += 1
                        break
            catches.append(catchesByOption)


        # Pirates Capacity Combinations
        pirate_change = []
        pirates_changes = []


        for i, pirate in enumerate(pirates):
            if catches[i] == len(marines_changes) :
                pirate_change.append([(pirate[0] , pirate[1] ,2)])
                continue
            pirate_change.append([(pirates[i])])
        pirates_changes = list(itertools.product(*pirate_change))
        pirates_changes = list(set(pirates_changes))

        # All combinations
        parts = (tuple(pirates_changes),) + (tuple(treasures_changes),) + (tuple(marines_changes),)
        all_resulted_states = list(itertools.product(*parts))


        nested_list = [[list(inner_tuple) for inner_tuple in list(outer_tuple)] for outer_tuple in all_resulted_states]

        for j, option in enumerate(nested_list):
            option = list(option)
            for i, pirate in enumerate(option[0]):
                for k, marine in enumerate(option[2]):
                    if self.marines_info[k][1][marine[1]] == option[0][i][1]:
                        if option[0][i][2] != 2:
                            # Change the value of option[0][i][2] to 2
                            nested_list[j][0][i] = (option[0][i][0], option[0][i][1], 2)
                            # Append the modified option to new_tuple

        nested_tuples = tuple(tuple(tuple(inner_tuple) for inner_tuple in outer_tuple) for outer_tuple in all_resulted_states)

        return tuple(nested_tuples)

    def get_R(self, state, action, next_state):

        pirates_old = state[0]
        pirates_new = next_state[0]
        marines = next_state[2]
        total_reward = 0

        if action == "reset":
            total_reward = -2

        else:
            for i, act in enumerate(action):
                # Case 2
                if act[0] == "deposit":
                    total_reward += (2 - pirates_old[i][2]) * 4
                # Case 3
                for k, marine in enumerate(marines):
                    if self.marines_info[k][1][marine[1]] == pirates_new[i][1]:
                        total_reward -= 1
                        break

        return total_reward

    def transitions(self, state, action, next_state):

        if action == 'reset':
            return 1

        pirates = state[0]
        treasures = state[1]
        treasures_next_state = next_state[1]
        marines = state[2]
        marines_next_state = next_state[2]

        # Marine Movement
        marines_prob = {}
        for i, marine in enumerate(marines):
            # Only one location
            if len(self.marines_info[i][1]) == 1:
                marines_prob[marine[0]] = 1

            # On sides of the path
            elif ((marine[1] == 0) or (marine[1] == len(self.marines_info[i][1]) - 1)):
                marines_prob[marine[0]] = 1 / 2

            else:
                marines_prob[marine[0]] = (1 / 3)


        # Treasure movement
        treasures_prob = {}
        for i, treasure in enumerate(treasures):

            change_prob = self.treasures_info[i][2]
            num_of_locations = len(self.treasures_info[i][1])
            move_prob = (1 / num_of_locations) * change_prob
            my_location_count = 0

            for location in self.treasures_info[i][1]:
                if treasure[1] == location:
                    my_location_count += 1

            # Stay on same place
            if treasure[1] == treasures_next_state[i][1]:
                treasures_prob[treasure[0]] = (1 - change_prob) + my_location_count * move_prob
            else:
                treasures_prob[treasure[0]] = move_prob

        total_marine_prob = 1
        for marine in marines_prob:
            total_marine_prob *= marines_prob[marine]

        total_treasure_prob = 1
        for treasure in treasures_prob:
            total_treasure_prob *= treasures_prob[treasure]

        return total_marine_prob * total_treasure_prob

    def value_iteration(self):

        self.V = {state: 0 for state in self.states}
        keys = [s for s in self.states]
        max_actions_per_time = dict.fromkeys(keys, 0)
        epsilon = 1/100
        delta = 0

        while True:
            V_updating = {state: 0 for state in self.states}
            for state in self.states:
                max_value = -inf
                best_action = 0
                for action in self.actions(state):
                    value = 0
                    next_states = self.resultWithoutTime(state , action)
                    for next_state in next_states:
                        probability = self.transitions(state, action, next_state)
                        value += probability * (self.V[next_state] + self.get_R(state, action , next_state))
                    value *= self.gamma

                    if value > max_value:
                        max_value = value
                        best_action = action

                V_updating[state] = max_value
                max_actions_per_time[state] = best_action

            delta = infinite_norm_subtraction(self.V, V_updating)
            self.policy = max_actions_per_time

            if delta < epsilon :
                self.V = V_updating
                break

            self.V = V_updating

    def generate_states(self, initial):
        init_state = self.convert_to_my_state(initial)
        pirates = init_state[0]
        treasures = init_state[1]
        marines = init_state[2]

        pirate_possible_locations = []
        for i, row in enumerate(self.grid):
            for j in range(len(row)):
                if self.grid[i][j] != "I": pirate_possible_locations.append((i, j))

        pirates_possibilities = ()
        for pirate in pirates:
            pirate_poss = ()
            for poss in pirate_possible_locations:
                for i in range(3):
                    pirate_poss = pirate_poss + ((pirate[0], poss, i),)
            pirates_possibilities = pirates_possibilities + (pirate_poss,)

        treasures_possibilites = ()
        for i, treasure in enumerate(treasures):
            tres_poss = ()
            tres_poss = tres_poss + ((treasure[0], treasure[1]),)
            for loc in self.treasures_info[i][1]:
                tres_poss = tres_poss + ((treasure[0], loc),)
            tres_poss = set(tres_poss)
            treasures_possibilites = treasures_possibilites + (tres_poss,)

        marines_possibilites = ()
        for i, marine in enumerate(marines):
            marine_poss = ()
            for j in range(len(self.marines_info[i][1])):
                marine_poss = marine_poss + ((marine[0], j),)
            marines_possibilites = marines_possibilites + (marine_poss,)

        pirates_combinations = list(itertools.product(*pirates_possibilities))
        treasures_combinations = list(itertools.product(*treasures_possibilites))
        marines_combinations = list(itertools.product(*marines_possibilites))

        combined_lists = []
        combined_lists.append(pirates_combinations)
        combined_lists.append(treasures_combinations)
        combined_lists.append(marines_combinations)

        all_combinations = list(itertools.product(*combined_lists))



        return all_combinations

    def act(self, state):

        new_state = self.convert_to_my_state(state)
        action = self.policy[new_state]

        if action == None:
            action = 'terminate'

        return action


def infinite_norm_subtraction(dict1, dict2):
    max_diff = 0
    for key in dict1.keys():
        if key in dict2:
            diff = abs(dict1[key] - dict2[key])
            if diff > max_diff:
                max_diff = diff
        else:
            diff = abs(dict1[key])
            if diff > max_diff:
                max_diff = diff

    for key in dict2.keys():
        if key not in dict1:
            diff = abs(dict2[key])
            if diff > max_diff:
                max_diff = diff

    return max_diff
