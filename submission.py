from environment import Player, GameState, GameAction, get_next_state
from utils import get_fitness
import numpy as np
from enum import Enum
import time

def heuristic(state: GameState, player_index: int) -> float:
    """
    Computes the heuristic value for the agent with player_index at the given state
    :param state:
    :param player_index: integer. represents the identity of the player. this is the index of the agent's snake in the
    state.snakes array as well.
    :return:
    """
    # Insert your code here...
    snake = state.snakes[player_index]
    d_min=0
    head_tail_factor=0.05
    if state.fruits_locations.__len__() is not 0:
        d_min = min(abs(d[0] - snake.head[0]) + abs(d[1] - snake.head[1]) for d in state.fruits_locations)
    d_head_tail = abs(snake.head[0] - snake.tail_position[0]) + abs(snake.head[1] - snake.tail_position[1])
    dont_die = 1 if snake.alive else 0
    # max(state.board_size[0], state.board_size[1]) -
    x=1
    if d_min>0:
        x=1/d_min
    return (snake.length + x + d_head_tail*head_tail_factor) * dont_die
    pass
    # + d_head_tail


class MinimaxAgent(Player):
    """
    This class implements the Minimax algorithm.
    Since this algorithm needs the game to have defined turns, we will model these turns ourselves.
    Use 'TurnBasedGameState' to wrap the given state at the 'get_action' method.
    hint: use the 'agent_action' property to determine if it's the agents turn or the opponents' turn. You can pass
    'None' value (without quotes) to indicate that your agent haven't picked an action yet.
    """

    class Turn(Enum):
        AGENT_TURN = 'AGENT_TURN'
        OPPONENTS_TURN = 'OPPONENTS_TURN'

    class TurnBasedGameState:
        """
        This class is a wrapper class for a GameState. It holds the action of our agent as well, so we can model turns
        in the game (set agent_action=None to indicate that our agent has yet to pick an action).
        """

        def __init__(self, game_state: GameState, agent_action: GameAction):
            self.game_state = game_state
            self.agent_action = agent_action

        @property
        def turn(self):
            return MinimaxAgent.Turn.AGENT_TURN if self.agent_action is None else MinimaxAgent.Turn.OPPONENTS_TURN

    def minimax(self, state: TurnBasedGameState, depth: int) -> float:
        if state.game_state.is_terminal_state or depth > 3 or state.game_state.get_possible_actions(player_index=self.player_index).__len__ is 0:
            return heuristic(state.game_state, self.player_index)
        if state.turn == self.Turn.OPPONENTS_TURN:
            curr_min = np.inf
            for opponents_actions in state.game_state.get_possible_actions_dicts_given_action(state.agent_action,
                                                                                              player_index=self.player_index):
                opponents_actions[self.player_index] = state.agent_action
                next_state = get_next_state(state.game_state, opponents_actions)
                our_turn = self.TurnBasedGameState(next_state, None)
                curr_min = min(self.minimax(our_turn, depth+1), curr_min)
            #print(curr_max)
            return curr_min
        else:
            curr_max = -np.inf
            for agent_action in state.game_state.get_possible_actions(player_index=self.player_index):
                opp_turn = self.TurnBasedGameState(state.game_state, agent_action)
                curr_max = max(self.minimax(opp_turn, depth), curr_max)
            return curr_max

    def get_action(self, state: GameState) -> GameAction:
        # Insert your code here...
        start = time.time()
        choose_max = -np.inf

        max_action = GameAction.LEFT
        for agent_action in state.get_possible_actions(player_index=self.player_index):
            head_tree = self.TurnBasedGameState(state, agent_action) #possible opponent action
            current_action_max = self.minimax(head_tree, 1)

            #print(choose_max, curr_result, "\n")
            if choose_max < current_action_max:
                choose_max = current_action_max
                max_action = agent_action
        end = time.time()
        self.counter_steps += 1
        self.avg_time = ((end-start)+self.avg_time*(self.counter_steps-1))/self.counter_steps
        return max_action
        pass


class AlphaBetaAgent(MinimaxAgent):

    def minimax(self, state: MinimaxAgent.TurnBasedGameState, depth: int, alpha=-np.inf, beta=np.inf) -> float:
        if state.game_state.is_terminal_state or depth > 5 or state.game_state.get_possible_actions(
                player_index=self.player_index).__len__ is 0:
            return heuristic(state.game_state, self.player_index)
        if state.turn == self.Turn.OPPONENTS_TURN:
            curr_min = np.inf
            for opponents_actions in state.game_state.get_possible_actions_dicts_given_action(state.agent_action,
                                                                                              player_index=self.player_index):
                opponents_actions[self.player_index] = state.agent_action
                next_state = get_next_state(state.game_state, opponents_actions)
                our_turn = self.TurnBasedGameState(next_state, None)
                curr_min = min(self.minimax(our_turn, depth + 1, alpha, beta), curr_min)
                beta = min(beta, curr_min)
                if curr_min <= alpha:
                    return -np.inf
            # print(curr_max)
            return curr_min
        else:
            curr_max = -np.inf
            for agent_action in state.game_state.get_possible_actions(player_index=self.player_index):
                opp_turn = self.TurnBasedGameState(state.game_state, agent_action)
                curr_max = max(self.minimax(opp_turn, depth, alpha, beta), curr_max)
                alpha = max(curr_max, alpha)
                if curr_max >= beta:
                    return np.inf
            return curr_max

    def get_action(self, state: GameState) -> GameAction:
        # Insert your code here...
        start = time.time()
        choose_max = -np.inf

        max_action = GameAction.LEFT
        for agent_action in state.get_possible_actions(player_index=self.player_index):
            head_tree = self.TurnBasedGameState(state, agent_action)  # possible opponent action
            current_action_max = self.minimax(head_tree, 1)

            # print(choose_max, curr_result, "\n")
            if choose_max < current_action_max:
                choose_max = current_action_max
                max_action = agent_action
                alpha = choose_max
        end = time.time()
        self.counter_steps += 1
        self.avg_time = ((end - start) + self.avg_time * (self.counter_steps - 1)) / self.counter_steps
        return max_action
        pass

def int2GA(x: int):
    if x == 0:
        return GameAction.LEFT
    if x == 1:
        return GameAction.STRAIGHT
    if x == 2:
        return GameAction.RIGHT

def SAHC_sideways():
    """
    Implement Steepest Ascent Hill Climbing with Sideways Steps Here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.

    an outline of the algorithm can be
    1) pick an initial state
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """
    action_vector = tuple(int2GA(np.random.choice(2)) for i in range(50))
    max_value = get_fitness(action_vector)
    for i in range(100):
        curr_vector = tuple(int2GA(np.random.choice(2)) for i in range(50))
        curr_value = get_fitness(curr_vector)
        if curr_value > max_value:
            action_vector = curr_vector
            max_value = curr_value
    print(action_vector)
    print(max_value)
    print("-----------------------------------------------------------")
    #action_vector = tuple(int2GA(i) for i in np.random.choice(3, 50))
    #print(action_vector)
    #print(get_fitness(action_vector))
    side_steps = 0
    action_set = {GameAction.LEFT, GameAction.STRAIGHT, GameAction.RIGHT}
    for i in range(50):
        curr_value = get_fitness(action_vector)
        operator = list(action_set.difference({action_vector[i]}))
        help_list = list(action_vector)
        max = -np.inf
        best_states = []
        for k in operator:
            help_list[i] = k
            new_state = tuple(help_list)
            new_state_value = get_fitness(new_state)
            if new_state_value > max:
                max = new_state_value
                best_states = [new_state]
            elif new_state_value == max:
                best_states.append(new_state)
        index = np.random.choice(best_states.__len__())
        if max > curr_value:
            action_vector = tuple(best_states[index])
            side_steps = 0
        elif max == curr_value and side_steps < 50:
            action_vector = tuple(best_states[index])
            print("side step")
            side_steps += 1
        else:
            print(action_vector)
            get_fitness(action_vector)
            print(curr_value)
            break
    pass

def flip_coin(x: float):
    c = np.random.random()
    return True if c < x else False

def local_search():
    """
    Implement your own local search algorithm here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.

    an outline of the algorithm can be
    1) pick an initial state/states
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """
    T_factor = 0.66
    for j in range(1):
        best_value_ever = -np.inf
        best_action_ever = []
# the local search algo for 10 times:
#-----------------------------------------------------------------------------------------------
        for k in range(10):
            action_vector = tuple(int2GA(i) for i in np.random.choice(3, 50))
            side_steps = 0
            action_set = {GameAction.LEFT, GameAction.STRAIGHT, GameAction.RIGHT}
            T = 10
            best_value = -np.inf
            best_action = []
# one iteration
# ____________________________________________________________________________________
            for i in range(50):
                curr_value = get_fitness(action_vector)
                operator = list(action_set.difference({action_vector[i]}))
                help_list = list(action_vector)
                max_ = -np.inf
                best_states = []
                for k in operator:
                    help_list[i] = k
                    new_state = tuple(help_list)
                    new_state_value = get_fitness(new_state)
                    if new_state_value > max_:
                        max_ = new_state_value
                        best_states = [new_state]
                    elif new_state_value == max_:
                        best_states.append(new_state)
                index = np.random.choice(best_states.__len__())
                if max_ > curr_value:
                    action_vector = tuple(best_states[index])
                    curr_value = max_
                    side_steps = 0
                elif max_ == curr_value and side_steps < 50:
                    action_vector = tuple(best_states[index])
                    side_steps += 1
                    curr_value = max_
                elif max_ < curr_value and flip_coin(np.exp(-abs(max_-curr_value)/T)):
                    action_vector = tuple(best_states[index])
                    side_steps = 0
                else:
                    break
                if best_value<curr_value:
                    best_action = action_vector
                    best_value = curr_value
                T = T * T_factor
# ____________________________________________________________________________________
            if best_value > best_value_ever:
                best_value_ever = best_value
                best_action_ever = best_action
        print(best_action_ever)
        print(best_value_ever)
        print(get_fitness(best_action_ever))
        T_factor *= 0.95

    pass


class TournamentAgent(Player):

    def get_action(self, state: GameState) -> GameAction:
        pass


if __name__ == '__main__':
    #SAHC_sideways()
    local_search()
