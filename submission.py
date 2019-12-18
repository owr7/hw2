from environment import Player, GameState, GameAction, get_next_state
from utils import get_fitness
import numpy as np
from enum import Enum


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

    def U(self, state: GameState):
        our_snake = state.snakes[self.player_index]
        grade = sum(1 for s in state.snakes if s.index is not self.player_index and s.length > our_snake.length)
        return -grade if grade > 0 else 1

    def minimax(self, state: TurnBasedGameState, depth: int) -> float:
        if state.game_state.is_terminal_state or depth > 0 or state.game_state.get_possible_actions(player_index=self.player_index).__len__ is 0:
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
        choose_max = -np.inf

        max_action = GameAction.LEFT
        for agent_action in state.get_possible_actions(player_index=self.player_index):
            head_tree = self.TurnBasedGameState(state, agent_action) #possible opponent action
            current_action_max = self.minimax(head_tree, 1)

            #print(choose_max, curr_result, "\n")
            if choose_max < current_action_max:
                choose_max = current_action_max
                max_action = agent_action
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
        return max_action
        pass


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
    pass


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
    pass


class TournamentAgent(Player):

    def get_action(self, state: GameState) -> GameAction:
        pass


if __name__ == '__main__':
    SAHC_sideways()
    local_search()
