from player import Player
from game_data import *

from typing import List, Dict, Tuple
import numpy as np

class HeuristicPlayer(Player):
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
        return legal_moves[0]

    def respond(self, legal_responses: List[Response]) -> Response:
        return Response.CALL_BS

    def block_respond(self) -> BlockResponse:
        return BlockResponse.CALL_BS

    def get_observation(self, obs: Observation) -> None:
        pass

    def flip_role(self) -> int:
        return np.random.choice([i for i in range(len(self.roles)) if self.roles[i] != Role.NONE])

class AdvancedHeuristicPlayer(Player):
    def __init__(self,
                 coins: int,
                 roles: List[Role],
                 num_players: int,
                 idx: int,
                 all_player_types: List[type],
                 all_params: List[Dict],
                 **kwargs):
        super().__init__(coins, roles, num_players, idx, all_player_types, all_params)
        self.last_move = None
    
    def try_move(self, move_type: Move, legal_moves: List[Tuple[Move, int]]):
        good_moves = [a for a in legal_moves if a[0] == move_type]
        if len(good_moves) > 0:
            return good_moves[np.random.choice(len(good_moves))]
        return None

    def move(self, legal_moves: List[Tuple[Move, int]]) -> Tuple[Move, int]:
        if Role.DUKE in self.roles or np.random.random() < .1:
            a = self.try_move(Move.TAX, legal_moves)
            if a is not None:
                return a
        if Role.CAPTAIN in self.roles or np.random.random() < .1:
            a = self.try_move(Move.STEAL, legal_moves)
            if a is not None:
                return a
        if Role.ASSASSIN in self.roles or np.random.random() < .1:
            a = self.try_move(Move.ASSASSINATE, legal_moves)
            if a is not None:
                return a
        if Role.AMBASSADOR in self.roles and np.random.random() > .8:
            a = self.try_move(Move.SWAP_CARDS, legal_moves)
            if a is not None:
                return a
        return legal_moves[0]

    def respond(self, legal_responses: List[Response]) -> Response:
        if self.last_move.obs == Move.TAX:
            if Role.DUKE in self.roles or np.random.random() < 0.1:
                return Response.BLOCK
        if (self.last_move.player_against + self.last_move.player) % self.num_players != self.idx:
            return Response.NOTHING
        if self.last_move.obs == Move.ASSASSINATE:
            if Role.CONTESSA in self.roles or np.random.random() < 0.4:
                return Response.BLOCK
            if np.random.random() < 0.2 or Role.NONE in self.roles:
                return Response.CALL_BS
        if self.last_move.obs == Move.STEAL:
            if Role.AMBASSADOR in self.roles or Role.CAPTAIN in self.roles or np.random.random() < 0.2:
                return Response.BLOCK
            if np.random.random() < 0.1:
                return Response.CALL_BS
        return Response.NOTHING

    def block_respond(self) -> BlockResponse:
        if np.random.random() < 0.2:
            return BlockResponse.CALL_BS
        return BlockResponse.NOTHING

    def get_observation(self, obs: Observation) -> None:
        if obs.obs_type == ObservationType.MOVE:
            self.last_move = obs

    def flip_role(self) -> int:
        return np.random.choice([i for i in range(len(self.roles)) if self.roles[i] != Role.NONE])