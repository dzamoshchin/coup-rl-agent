from typing import List, Tuple, Dict
from game_data import *
import numpy as np
from player import Player
from simulator import Simulator
from player import *
from copy import deepcopy
from q import QPlayer
import timeit


class MCTSPlayer(QPlayer):
    def __init__(self,
                 coins: int,
                 roles: List[Role],
                 num_players: int,
                 idx: int,
                 all_player_types: List[type],
                 all_params: List[Dict],
                 Q: Dict, N: Dict, c: int, alpha: float = 0.01, depth: int = 10, num_simulations: int = 10):
        super().__init__(coins, roles, num_players, idx, all_player_types, all_params, Q, N, c, alpha)

        self.gamma = 1

        self.depth = depth
        self.num_simulations = num_simulations

        self.exact_game_state = [2 for _ in range(num_players)] + [len(roles) for _ in range(num_players)] + sorted(roles)
        
        self.roles_per_player = len(self.roles)


    def UCB1(self, s, a):
        if self.N[s][a] == 0:
            return 1e4
        return self.Q[s][a] + self.c * np.sqrt(np.log(sum(self.N[s][ap] for ap in self.N[s])) / (1e-4 + self.N[s][a]))

    def simulate(self, sim: Simulator, depth: int):
        '''if depth <= 0:
            return
        
        legal_moves = sim.get_legal_moves(sim.players[self.idx])
        history = tuple(sim.players[self.idx].cur_game_state) + (0,)
        
        if history not in self.N and legal_moves[0][0] not in self.N[history]:
            for move in legal_moves:
                self.Q[history][move[0]] = 0.0
                self.N[history][move[0]] = 0
            return

        sim.take_turn(action)
        while (sim.cur_turn != self.idx):
            if sim.run_game(1) != -1:
                return
        self.N[history][action] += 1
        return '''
        # start_time = timeit.default_timer()
        visible_state, game_over = sim.run_game(depth)
        # print(timeit.default_timer() - start_time)
        if game_over:
            if visible_state == self.idx:
                utility = 1
            else:
                utility = -1
        else:
            final_game_state = [coin_bucket(c) for c in visible_state.coin_list] + \
                                            [num_roles for num_roles in visible_state.num_roles_list] + \
                                            sorted(self.roles)
            
            history = tuple(final_game_state) + (0,)
            if len(self.Q[history]) == 0:
                utility = 0
            else:
                next_action = max(self.Q[history], key=lambda a: self.Q[history][a])
                utility = self.Q[history][next_action]
        
        return utility




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
                                    [Role.NONE for _ in range(2-self.exact_game_state[self.num_players + i])])
            players.append(self.all_player_types[i](self.coins, player_roles[-1], self.num_players, i,
                                                    new_player_types, self.all_params, **self.all_params[i]))

        sample = {'verbosity': 0, 'cur_turn': self.idx, 
                  'alive_players': [self.exact_game_state[i] > 0 for i in range(self.num_players, 2 * self.num_players)],
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
                self.Q[history][action] = (1-self.alpha) * self.Q[history][action] + self.alpha
        else:
            for history, action in self.past_ha:
                self.Q[history][action] = (1-self.alpha) * self.Q[history][action] - self.alpha


def coin_bucket(c):
            # if c < 3:
            #     return 0
            if c < 7:
                return 1
            # if c < 10:
            #     return 2
            return 3