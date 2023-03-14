from typing import List, Tuple, Dict
from game_data import *
import numpy as np
from player import Player
from collections import defaultdict


class MCTSPlayer(Player):
    def __init__(self, Q: Dict[int, Dict[int, int]], N: Dict[int, Dict[int, int]], c: int, depth: int, num_simulations: int):
        super().__init__()

        self.c = c # exploration constant
        
        # should they be initialized to zero on every run?
        # self.Q = defaultdict(lambda: defaultdict(int)) # action value estimates
        # self.N = defaultdict(lambda: defaultdict(int)) # visit counts

        self.Q = Q
        self.N = N

        self.U = defaultdict()

        self.depth = depth
        self.num_simulations = num_simulations
    
    def assign(self, coins: int, roles: List[Role], num_players: int, idx: int):
        super().assign(coins, roles, num_players, idx)
    
    def get_history_from_state(self, players: List[Role]) -> Tuple[Tuple[int, int]]:
        history = []
        for p in players:
            num_lives = sum(1 for role in p.roles if role != Role.NONE)
            history.append((num_lives, p.coins))
        return tuple(history)

    # assumes that the opponent will never block, but what we really want is to model belief of opponent's roles/ability to block actions like stealing/foreign aid
    def TR(self, history: Tuple[Tuple[int, int]], action: Move, idx_against: int) -> Tuple[int, Tuple[Tuple[int, int]]]:

        my_cards, my_coins = history[self.idx]
        ag_cards, ag_coins = history[idx_against]

        reward = 0
        if action == Move.INCOME:
            # reward = 1
            my_coins += 1
        elif action == Move.FOREIGN_AID:
            # reward = 2
            my_coins += 2
        elif action == Move.COUP:
            my_coins -= 7
            ag_cards -= 1
            reward = 100 if ag_cards == 0 else 30
        elif action == Move.ASSASSINATE:
            my_coins -=3
            ag_cards -= 1
            # reward = 100 if ag_cards == 0 else 30
        elif action == Move.STEAL:
            # reward = 2
            my_coins += 2
            ag_coins -= 2
        elif action == Move.TAX:
            # reward = 3
            my_coins += 3

        next_history = []
        for index, player in enumerate(history):
            if index == self.idx:
                next_history.append((my_cards, my_coins))
            elif index == idx_against:
                next_history.append((ag_cards, ag_coins))
            else:
                next_history.append(player)
        next_history = tuple(next_history)

        return reward, next_history

    def bonus(self, Nsa: int, Ns: int) -> float:
        return float("inf") if Nsa == 0 else np.sqrt(np.log(Ns)/Nsa)

    def explore(self, history: Tuple[Tuple[int, int]], legal_moves: List[Tuple[Move, int]]) -> Tuple[Move, int]:
        Nh = sum(self.N[history][a[0]] for a in legal_moves)
        return max(legal_moves, key = lambda a: self.Q[history][a[0]] + self.c * self.bonus(self.N[history][a[0]], Nh))

    def simulation(self, legal_moves: List[Tuple[Move, int]], history: Tuple[Tuple[int, int]], depth: int, gamma: int = 1) -> int:
        if (depth <= 0):
            return 1.0 # TODO: replace with actual U estimate
        
        if history not in self.N and legal_moves[0][0] not in self.N[history]:
            for move in legal_moves:
                self.N[history][move[0]] = 0
                self.N[history][move[0]] = 0
            return 1.0 # TODO: replace with actual U estimate
        
        action, against = self.explore(history, legal_moves)
        reward, new_history = self.TR(history, action, against)

        # this is wrong, we need to update legal_moves after taking a simulation step
        q = reward + gamma * self.simulation(legal_moves, new_history, depth - 1)
        self.N[history][action] += 1
        self.Q[history][action] += (q - self.Q[history][action]) / self.N[history][action] 
        return q

    def move(self, legal_moves: List[Tuple[Move, int]], player_states: List[type]) -> Tuple[Move, int]:
        history = self.get_history_from_state(player_states)

        for _ in range(self.num_simulations):
            self.simulation(legal_moves, history, self.depth)

        return max(legal_moves, key = lambda a: self.Q[history][a[0]])

    # TODO: use MCTS for responses/blocks
    def respond(self, legal_responses: List[Response]) -> Response:
        return np.random.choice(legal_responses)

    def block_respond(self) -> BlockResponse:
        return np.random.choice(BlockResponse)

    def get_observation(self, obs: Observation) -> None:
        pass

    def flip_role(self) -> int:
        return np.random.choice([i for i in range(len(self.roles)) if self.roles[i] != Role.NONE])
