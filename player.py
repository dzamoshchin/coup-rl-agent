from typing import List, Tuple, Dict
from game_data import *
import numpy as np


class Player:

    def __init__(self,
                 coins: int,
                 roles: List[Role],
                 num_players: int,
                 idx: int,
                 all_player_types: List[type],
                 all_params: List[Dict],
                 *kwargs):
        self.coins = coins
        self.roles = roles
        self.num_players = num_players
        self.idx = idx
        self.all_player_types = all_player_types
        self.all_params = all_params

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
    
    def game_over(self, won) -> None:
        pass


class RandomPlayer(Player):
    def __init__(self,
                 coins: int,
                 roles: List[Role],
                 num_players: int,
                 idx: int,
                 all_player_types: List[type],
                 all_params: List[Dict],
                 **kwargs):
        super().__init__(coins, roles, num_players, idx, all_player_types, all_params)

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
    
class HeuristicPlayer(Player):
    def __init__(self,
                 coins: int,
                 roles: List[Role],
                 num_players: int,
                 idx: int,
                 all_player_types: List[type],
                 all_params: List[Dict]):
        super().__init__(coins, roles, num_players, idx, all_player_types, all_params)
    
    def move(self, legal_moves: List[Tuple[Move, int]]) -> Tuple[Move, int]:
        return legal_moves[0]

    def respond(self, legal_responses: List[Response]) -> Response:
        return Response.CALL_BS

    def block_respond(self) -> BlockResponse:
        return BlockResponse.CALL_BS

    def get_observation(self, obs: Observation) -> None:
        pass

    def flip_role(self) -> int:
        return np.random.choice([i for i in range(len(self.roles)) if self.roles[i] != Role.NONE])