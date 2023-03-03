from typing import List, Tuple
from game_data import *
import numpy as np


class Player:

    def __init__(self,
                 coins: int,
                 roles: List[Role],
                 num_players: int,
                 idx: int):
        self.coins = coins
        self.roles = roles
        self.idx = idx

    def move(self, legal_moves: List[Tuple[Move, int]]) -> Tuple[Move, int]:
        pass

    def respond(self, legal_responses: List[Response]) -> Response:
        pass

    def block_respond(self) -> BlockResponse:
        pass

    def get_observation(self, obs: Observation) -> None:
        pass

    def flip_role(self) -> int:
        pass


class RandomPlayer(Player):
    def __init__(self, coins: int, roles: List[Role], num_players: int, idx: int):
        super().__init__(coins, roles, num_players, idx)

    def move(self, legal_moves: List[Tuple[Move, int]]) -> Tuple[Move, int]:
        idx = np.random.randint(len(legal_moves))
        return legal_moves[idx]

    def respond(self, legal_responses: List[Response]) -> Response:
        return np.random.choice(legal_responses)

    def block_respond(self) -> BlockResponse:
        return np.random.choice(BlockResponse)

    def get_observation(self, obs: Observation) -> None:
        pass

    def flip_role(self) -> int:
        return np.random.choice([i for i in range(len(self.roles)) if self.roles[i] != Role.NONE])
