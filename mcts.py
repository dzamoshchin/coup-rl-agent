from typing import List, Tuple, Dict
from game_data import *
import numpy as np
from player import Player
from simulator import Simulator
from player import *
import copy
import timeit


class MCTSPlayer(Player):
    def __init__(self, coins: int, roles: List[Role], num_players: int, idx: int,
                 Q: Dict, N: Dict, c: int, depth: int, num_simulations: int, alpha: float = 0.01):
        super().__init__(coins, roles, num_players, idx)

        self.c = c  # exploration constant
        self.alpha = alpha  # learning rate
        self.gamma = 1

        self.Q = Q
        self.N = N

        self.depth = depth
        self.num_simulations = num_simulations

        self.cur_game_state = [0 for _ in range(num_players)] + [len(roles) for _ in range(num_players)] + roles
        self.past_ha = []

    def UCB1(self, s, a):
        if self.N[s][a] == 0:
            return 1e4
        return self.Q[s][a] + self.c * np.sqrt(np.log(sum(self.N[s][ap] for ap in self.N[s])) / (1e-4 + self.N[s][a]))

    def create_simulator(self, game_state: Dict) -> Simulator:
        for idx, player_type in enumerate(game_state["player_types"]):
            if player_type == MCTSPlayer:
                game_state["player_types"][idx] = DummyMCTSPlayer

        # replace MCTSPlayers with DummyMCTSPlayers
        for idx, player in enumerate(game_state["players"]):
            if isinstance(player, MCTSPlayer):
                game_state["players"][idx] = DummyMCTSPlayer(player.coins, player.roles, player.num_players, player.idx, player.Q,
                                                             player.N, player.c, player.depth, player.num_simulations, player.alpha)

        return Simulator(**game_state)
    
    def simulate(self, sim: Simulator, depth: int):
        if depth <= 0:
            return
        
        legal_moves = sim.get_legal_moves(sim.players[self.idx])
        history = tuple(sim.players[self.idx].cur_game_state) + (0,)
        
        if history not in self.N and legal_moves[0][0] not in self.N[history]:
            for move in legal_moves:
                self.Q[history][move[0]] = 0.0
                self.N[history][move[0]] = 0
            return

        action = max(legal_moves, key=lambda a: self.UCB1(history, a))
        sim.take_turn(action)
        while (sim.cur_turn != self.idx):
            if sim.run_game(1) != -1:
                return
        self.N[history][action] += 1

    def move(self, legal_moves: List[Tuple[Move, int]], game_state: Dict, dummy: bool = False) -> Tuple[Move, int]:
        history = tuple(self.cur_game_state) + (0,)

        if not dummy:
            for _ in range(self.num_simulations):
                # starttime = timeit.default_timer()
                new_game = copy.deepcopy(game_state) # extremely extremely expensive, is there an alternative?
                # print("time", timeit.default_timer() - starttime)
                sim = self.create_simulator(new_game)
                self.simulate(sim, self.depth)

        action = max(legal_moves, key=lambda a: self.UCB1(history, a))
        self.past_ha.append([history, action])
        self.N[history][action] += 1
        return action

    def respond(self, legal_responses: List[Response]) -> Response:
        history = tuple(self.cur_game_state) + (0,)
        a = max(legal_responses, key=lambda a: self.UCB1(history, a))
        self.past_ha.append([history, a])
        self.N[history][a] += 1
        return a

    def block_respond(self) -> BlockResponse:
        history = tuple(self.cur_game_state) + (0,)
        a = max([BlockResponse.NOTHING, BlockResponse.CALL_BS], key=lambda a: self.UCB1(history, a))
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

        self.cur_game_state = [coin_bucket(c) for c in obs.visible_state.coin_list] + \
                              [num_roles for num_roles in obs.visible_state.num_roles_list] + \
                              sorted(self.roles)

    def flip_role(self) -> int:
        return np.random.choice([i for i in range(len(self.roles)) if self.roles[i] != Role.NONE])

    def game_over(self, won) -> None:
        if won:
            for history, action in self.past_ha:
                self.Q[history][action] = (1-self.alpha) * self.Q[history][action] + self.alpha
        else:
            for history, action in self.past_ha:
                self.Q[history][action] = (1-self.alpha) * self.Q[history][action] - self.alpha

class DummyMCTSPlayer(MCTSPlayer):
    def __init__(self, coins: int, roles: List[Role], num_players: int, idx: int,
                 Q: Dict, N: Dict, c: int, depth: int, num_simulations: int, alpha: float = 0.01):
         super().__init__(coins, roles, num_players, idx, Q, N, c, depth, num_simulations, alpha)

    def move(self, legal_moves: List[Tuple[Move, int]], game_state: Dict) -> Tuple[Move, int]:
        return super().move(legal_moves, game_state, True)

    def respond(self, legal_responses: List[Response]) -> Response:
        return super().respond(legal_responses)

    def block_respond(self) -> BlockResponse:
        return super().block_respond()

    def get_observation(self, obs: Observation) -> None:
        return super().get_observation(obs)

    def flip_role(self) -> int:
        return super().flip_role()
    
    def game_over(self, won) -> None:
        return super().game_over(won)