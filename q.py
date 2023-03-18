from typing import List, Tuple, Dict
from game_data import *
import numpy as np
from player import Player
from copy import deepcopy


class QPlayer(Player):
    def __init__(self,
                 coins: int,
                 roles: List[Role],
                 num_players: int,
                 idx: int,
                 all_player_types: List[type],
                 all_params: List[Dict],
                 Q: Dict, N: Dict, c: int, alpha: float = 0.01, learn: bool = True, **kwargs):
        super().__init__(coins, roles, num_players, idx, all_player_types, all_params)

        self.c = c  # exploration constant
        self.alpha = alpha  # learning rate

        self.Q = Q
        self.N = N

        self.cur_game_state = [1 for _ in range(num_players)] + [len(roles) for _ in range(num_players)] + sorted(roles)
        self.past_ha = []

        self.learn = learn
        self.last_action = None

        def __deepcopy__(self, memo=None):
            return self(self.coins, deepcopy(self.roles),
                          self.num_players,
                          self.idx,
                          self.all_player_types,
                          self.all_params,
                          self.Q,
                          self.N, self.c, self.depth, self.num_simulations, self.alpha)

    def UCB1(self, s, a):
        if self.N[s][a] == 0:
            return 1e4
        return self.Q[s][a] + self.c * np.sqrt(np.log(sum(self.N[s][ap] for ap in self.N[s])) / (1e-4 + self.N[s][a]))

    def move(self, legal_moves: List[Tuple[Move, int]]) -> Tuple[Move, int]:
        history = tuple(self.cur_game_state) + (0,)
        a = max(legal_moves, key=lambda a: self.UCB1(history, a))
        if self.learn:
            self.past_ha.append([history, a])
            self.N[history][a] += 1
        return a

    def respond(self, legal_responses: List[Response]) -> Response:
        history = tuple(self.cur_game_state) + (1, self.last_action)
        a = max(legal_responses, key=lambda a: self.UCB1(history, a))
        if self.learn:
            self.past_ha.append([history, a])
            self.N[history][a] += 1
        return a

    def block_respond(self) -> BlockResponse:
        history = tuple(self.cur_game_state) + (2, self.last_action)
        a = max([BlockResponse.NOTHING, BlockResponse.CALL_BS], key=lambda a: self.UCB1(history, a))
        if self.learn:
            self.past_ha.append([history, a])
            self.N[history][a] += 1
        return a

    def get_observation(self, obs: Observation) -> None:
        def coin_bucket(c):
            # if c < 3:
            #     return 0
            if c < 7:
                return 1
            # if c < 10:
            #     return 2
            return 3
        
        start_on_idx = lambda l: l[self.idx:] + l[:self.idx]

        self.cur_game_state = start_on_idx([coin_bucket(c) for c in obs.visible_state.coin_list]) + \
                              start_on_idx([num_roles for num_roles in obs.visible_state.num_roles_list]) + \
                              sorted(self.roles)
        self.last_action = (obs.obs_type, obs.obs)

    def flip_role(self) -> int:
        return np.random.choice([i for i in range(len(self.roles)) if self.roles[i] != Role.NONE])

    def game_over(self, won) -> None:
        if won:
            for history, action in self.past_ha:
                self.Q[history][action] = (1-self.alpha) * self.Q[history][action] + self.alpha
        else:
            for history, action in self.past_ha:
                self.Q[history][action] = (1-self.alpha) * self.Q[history][action] - self.alpha
