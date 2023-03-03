from enum import IntEnum
from typing import List


class Role(IntEnum):
    AMBASSADOR = 0
    ASSASSIN = 1
    CAPTAIN = 2
    CONTESSA = 3
    DUKE = 4
    NONE = 5


class Action(IntEnum):
    pass


class Move(Action):
    INCOME = 0
    FOREIGN_AID = 1
    COUP = 2
    SWAP_CARDS = 3
    ASSASSINATE = 4
    STEAL = 5
    TAX = 6


class Response(Action):
    NOTHING = 0
    CALL_BS = 1
    BLOCK = 2


class BlockResponse(Action):
    NOTHING = 0
    CALL_BS = 1


class FlipCard(Action):
    AMBASSADOR = 0
    ASSASSIN = 1
    CAPTAIN = 2
    CONTESSA = 3
    DUKE = 4


class ObservationType(IntEnum):
    MOVE = 0
    RESPONSE = 1
    BLOCK_RESPONSE = 2
    FLIP_ROLE = 3
    SWAP_ROLE = 4


class Observation:
    def __init__(self,
                 obs_type: ObservationType,
                 obs: Action,
                 player: int,
                 player_against: int
                 ):
        self.obs_type = obs_type
        self.obs = obs
        self.player = player
        self.player_against = player_against

    def __str__(self):
        obs = [Move, Response, BlockResponse, FlipCard, FlipCard][self.obs_type](self.obs).name
        return f'Observation: {ObservationType(self.obs_type).name}, {obs}, Player {self.player} to {self.player_against}'


BLOCKABLE_MOVES = (Move.ASSASSINATE, Move.STEAL, Move.FOREIGN_AID)
BS_ABLE_MOVES = (Move.SWAP_CARDS, Move.ASSASSINATE, Move.STEAL, Move.TAX)


def get_proper_roles(move: Move, block: bool = False) -> List[Role]:
    if block:
        if move == Move.ASSASSINATE:
            return [Role.CONTESSA]
        elif move == Move.STEAL:
            return [Role.CAPTAIN, Role.AMBASSADOR]
        elif move == Move.FOREIGN_AID:
            return [Role.DUKE]
        else:
            return []
    else:
        if move == Move.TAX:
            return [Role.DUKE]
        elif move == Move.STEAL:
            return [Role.CAPTAIN]
        elif move == Move.ASSASSINATE:
            return [Role.ASSASSIN]
        elif move == Move.SWAP_CARDS:
            return [Role.AMBASSADOR]
        else:
            return []
