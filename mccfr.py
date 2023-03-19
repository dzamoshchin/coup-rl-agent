from typing import List, Tuple, Dict
from game_data import *
import numpy as np
from player import Player
from simulator import Simulator
from player import *
from copy import deepcopy
from q import QPlayer
import timeit


class MCCFRPlayer(Player):
    def __init__(self,
                 coins: int,
                 roles: List[Role],
                 num_players: int,
                 idx: int,
                 all_player_types: List[type],
                 all_params: List[Dict],
                 utilities,
                 regrets,
                 strat_profile,
                 eps: float = 0.05,
                 beta: float = 1e6,
                 tau: float = 1e3):
        super().__init__(coins, roles, num_players, idx, all_player_types, all_params)

        self.gamma = 1

        self.eps = eps
        self.beta = beta
        self.tau = tau
        self.utilities = utilities
        self.history = []
        self.roles_per_player = len(self.roles)

        self.regrets = regrets
        self.strat_profile = strat_profile

    def walk_tree(self, sim: Simulator, sample_prob, depth=100):
        if depth == 0 or sim.is_winner():
            # Game over; return utility
            if sim.check_winner() == self.idx:
                return 1 / sample_prob
            return -1 / sample_prob
        if sim.cur_turn != self.idx:
            for a in sim.get_legal_moves():
                self.strat_profile[tuple(self.cur_game_state) + (0,)][a]

    def move(self, legal_moves: List[Tuple[Move, int]]) -> Tuple[Move, int]:
        history = tuple(self.cur_game_state) + (0,)

        for _ in range(self.num_simulations):
            action = max(legal_moves, key=lambda a: self.UCB1(history, a))
            sim = Simulator(**self.sample_from_belief())
            utility = self.simulate(sim, self.depth)
            self.N[history][action] += 1
            self.Q[history][action] += (utility - self.Q[history][action]) / self.N[history][action]

        action = max(legal_moves, key=lambda a: self.UCB1(history, a))
        self.past_ha.append([history, action])
        return action

    def respond(self, legal_responses: List[Response]) -> Response:
        history = tuple(self.cur_game_state) + (1,)
        a = max(legal_responses, key=lambda a: self.UCB1(history, a))
        self.past_ha.append([history, a])
        self.N[history][a] += 1
        return a

    def block_respond(self) -> BlockResponse:
        history = tuple(self.cur_game_state) + (2,)
        a = max([BlockResponse.NOTHING, BlockResponse.CALL_BS], key=lambda a: self.UCB1(history, a))
        self.past_ha.append([history, a])
        self.N[history][a] += 1
        return a

    def get_observation(self, obs: Observation) -> None:
        self.cur_game_state = [coin_bucket(c) for c in obs.visible_state.coin_list] + \
                              [num_roles for num_roles in obs.visible_state.num_roles_list] + \
                              sorted(self.roles)
        self.exact_game_state = [c for c in obs.visible_state.coin_list] + \
                                [num_roles for num_roles in obs.visible_state.num_roles_list] + \
                                sorted(self.roles)
        self.history.append(obs.obs)
    
    def get_abstract_history(self) -> Tuple:
        pass

    def sample_from_belief(self):
        roles = [Role.AMBASSADOR, Role.ASSASSIN, Role.CAPTAIN, Role.CONTESSA, Role.DUKE] * 3
        middle_cards = [i for i in roles]
        np.random.shuffle(middle_cards)

        for role in self.roles:
            if role != Role.NONE:
                del middle_cards[middle_cards.index(role)]
        # Initialize players and give them their roles
        player_roles = []
        players = []
        new_player_types = self.all_player_types[:-1] + [QPlayer]
        for i in range(self.num_players):
            if i == self.idx:
                player_roles.append(deepcopy(self.roles))
            else:
                player_roles.append([middle_cards.pop(0) for _ in range(self.exact_game_state[self.num_players + i])] + \
                                    [Role.NONE for _ in range(2 - self.exact_game_state[self.num_players + i])])
            players.append(self.all_player_types[i](self.coins, player_roles[-1], self.num_players, i,
                                                    new_player_types, self.all_params, **self.all_params[i]))

        sample = {'verbosity': 0, 'cur_turn': self.idx,
                  'alive_players': [self.exact_game_state[i] > 0 for i in
                                    range(self.num_players, 2 * self.num_players)],
                  'roles_per_player': self.roles_per_player,
                  'player_types': new_player_types,
                  'params': self.all_params,
                  'roles': player_roles,
                  'middle_cards': middle_cards,
                  'players': players
                  }

        return sample

    def flip_role(self) -> int:
        return np.random.choice([i for i in range(len(self.roles)) if self.roles[i] != Role.NONE])

    def game_over(self, won) -> None:
        if won:
            for history, action in self.past_ha:
                self.Q[history][action] = (1 - self.alpha) * self.Q[history][action] + self.alpha
        else:
            for history, action in self.past_ha:
                self.Q[history][action] = (1 - self.alpha) * self.Q[history][action] - self.alpha


def coin_bucket(c):
    # if c < 3:
    #     return 0
    if c < 7:
        return 1
    # if c < 10:
    #     return 2
    return 3