from typing import List, Tuple, Dict
from game_data import *
from player import *
import numpy as np
from time import time
from collections import defaultdict
# from tqdm import trange
import pickle
from copy import deepcopy


class Simulator():
    def __init__(self, **kwargs):
        # The IDE doesn't like when I don't explicitly acknowledge these exist
        self.players: List[Player] = []
        self.roles_per_player: int = 0
        self.alive_players: List[bool] = []
        self.verbosity: int = 0
        self.middle_cards: List[Role] = []
        self.player_types: List[type] = []
        self.__dict__.update(kwargs)

    @classmethod
    def from_start(cls,
                   player_types: List[type],
                   params: List[Dict],
                   player_roles: List[List[Role]] = None,
                   coins: int = 2,
                   roles_per_player: int = 2,
                   verbosity: int = 0, ):

        cur_turn = 0
        alive_players = [True] * len(player_types)

        # Initialize standard deck of roles (3 of each role) and shuffle
        if player_roles is None:
            roles = [Role.AMBASSADOR, Role.ASSASSIN, Role.CAPTAIN, Role.CONTESSA, Role.DUKE] * 3
            middle_cards = [i for i in roles]
            np.random.shuffle(middle_cards)

            if len(player_types) * roles_per_player > len(roles):
                raise ValueError(
                    f"There are {len(player_types)} players and {roles_per_player} roles per player but just {len(roles)} roles.")

            # Initialize players and give them their roles
            players: List[Player] = []
            for idx, player_type in enumerate(player_types):
                player_roles = [middle_cards.pop(0) for _ in range(roles_per_player)]
                players.append(player_type(coins, player_roles, len(player_types), idx,
                                           player_types, params, **params[idx]))
        else:
            players: List[Player] = []
            middle_cards = [Role.AMBASSADOR, Role.ASSASSIN, Role.CAPTAIN, Role.CONTESSA, Role.DUKE] * 3
            np.random.shuffle(middle_cards)
            for idx, player in enumerate(player_roles):
                players.append(player_types[idx](coins, player_roles, len(player_types), idx,
                                                 player_types, params, **params[idx]))
                for role in player:
                    del middle_cards[middle_cards.index(role)]

        return cls(verbosity=verbosity, cur_turn=cur_turn, alive_players=alive_players,
                   roles_per_player=roles_per_player, player_types=player_types, roles=player_roles,
                   middle_cards=middle_cards, players=players)

    def check_winner(self) -> int:
        if self.is_winner():
            winner = self.alive_players.index(True)
            if self.verbosity > 2:
                print(f"Player {winner} wins")
            for i, player in enumerate(self.players):
                self.players[i].game_over(won=(i == winner))
            return winner

    def run_game(self, depth: int = None) -> int:
        if depth is None:
            while True:
                if self.is_winner():
                    return self.check_winner()
                self.take_turn()
        else:
            for _ in range(depth):
                if self.is_winner():
                    return self.check_winner()
                self.take_turn()

            return -1

    def is_winner(self):
        return self.alive_players.count(True) == 1

    def take_turn(self, action=None):
        # Print game state (optional)
        if self.verbosity > 2:
            self.print_game_state()
        # Player makes a move
        cur_player = self.players[self.cur_turn]
        legal_moves = self.get_legal_moves(cur_player)
        if action is not None:
            move, player_against = action
        else:
            move, player_against = cur_player.move(legal_moves)
        if (move, player_against) not in legal_moves:
            raise ValueError(f"Player of type {type(cur_player)} made illegal move {move} against {player_against}")

        # Other players observe the move
        obs = Observation(ObservationType.MOVE, move, self.cur_turn, player_against, None)
        self.send_observation(obs)

        # Players who assassinate pay 3 coins up front
        if move == Move.ASSASSINATE:
            if cur_player.coins < 3:
                self.kill_player(cur_player)
                if self.verbosity > 0:
                    print(f'Player of type {type(cur_player)} tried to assassinate with under 3 coins.')
                return
            cur_player.coins -= 3

        # Other players can respond to the move
        success = True
        for player in self.players[self.cur_turn + 1:] + self.players[:self.cur_turn]:
            legal_responses = self.get_legal_responses(player_against == player.idx, move)
            if len(legal_responses) == 1:
                break
            if not self.is_alive(player):
                continue

            response = player.respond(legal_responses)

            # Let players know
            obs = Observation(ObservationType.RESPONSE, response, player.idx, self.cur_turn)
            self.send_observation(obs)

            # Handle blocks and BS calls
            if response == Response.BLOCK:
                if move in BLOCKABLE_MOVES:
                    success = self.handle_block(cur_player, player, move)
                    break

            elif response == Response.CALL_BS:
                if move in BS_ABLE_MOVES:
                    success = self.handle_bs_call(cur_player, player, move, block=False)
                    break

        # Execute the move
        if success:
            self.make_move(cur_player, player_against, move)

        # Advance to next player
        self.cur_turn += 1
        self.cur_turn = self.cur_turn % len(self.players)
        while not self.alive_players[self.cur_turn]:
            self.cur_turn += 1
            self.cur_turn = self.cur_turn % len(self.players)

    def make_move(self, cur_player: Player, op_idx: int, move: Move) -> None:

        if move == Move.INCOME:
            cur_player.coins += 1
            return
        elif move == Move.FOREIGN_AID:
            cur_player.coins += 2
            return
        elif move == Move.TAX:
            cur_player.coins += 3
            return
        elif move == Move.SWAP_CARDS:
            old_roles = [role for role in cur_player.roles if role != Role.NONE]
            new_roles = [self.middle_cards.pop(0) for _ in range(len(old_roles))] + \
                        [Role.NONE for _ in range(self.roles_per_player - len(old_roles))]
            cur_player.roles = new_roles
            self.middle_cards += old_roles
            return

        other_player = self.players[op_idx]

        if move == Move.ASSASSINATE:
            if self.is_alive(other_player):
                self.lose_role(other_player)
        elif move == Move.COUP:
            if cur_player.coins < 7:
                self.kill_player(cur_player)
                if self.verbosity > 0:
                    print(f'Player of type {type(cur_player)} tried to coup with under 7 coins.')
            else:
                self.lose_role(other_player)
                cur_player.coins -= 7
        elif move == Move.STEAL:
            if other_player.coins == 0:
                self.kill_player(cur_player)
                if self.verbosity > 0:
                    print(f'Player of type {type(cur_player)} tried to steal from someone with no coins.')
            else:
                stolen_coins = min(2, other_player.coins)
                cur_player.coins += stolen_coins
                other_player.coins -= stolen_coins

    def handle_block(self, block_against: Player, blocker: Player, move: Move) -> bool:
        # Returns whether the action goes through (so True means block is UNsuccessful)
        cp_idx, op_idx = block_against.idx, blocker.idx

        # Get response to block
        block_response = block_against.block_respond()

        # Let players know
        obs = Observation(ObservationType.BLOCK_RESPONSE, block_response, cp_idx, op_idx)
        self.send_observation(obs)

        # Handle BS call on the block
        if block_response == BlockResponse.CALL_BS:
            return not self.handle_bs_call(blocker, block_against, move)
        return False

    def handle_bs_call(self, bs_against: Player, bs_caller: Player, move: Move, block: bool = False) -> bool:
        # Return whether action goes through (so True means the player having BS called against them was telling the truth)
        cp_idx, op_idx = bs_against.idx, bs_caller.idx
        proper_roles = get_proper_roles(move, block)
        for i, role in enumerate(bs_against.roles):
            if role in proper_roles:
                # BS Call was unsuccessful for caller
                # Swap the card out
                self.middle_cards.append(role)
                bs_against.roles[i] = self.middle_cards.pop(0)

                # Let players know this
                obs = Observation(ObservationType.SWAP_ROLE, FlipCard(role), cp_idx, cp_idx)
                self.send_observation(obs)

                # Player calling BS loses a role
                self.lose_role(bs_caller)
                return True

        # BS Call was successful
        # BS against loses a role
        self.lose_role(bs_against)
        return False

    def lose_role(self, player: Player):
        idx = player.idx
        flipped_role_idx = player.flip_role()
        flipped_role = player.roles[flipped_role_idx]
        if flipped_role == Role.NONE:
            if self.verbosity > 0:
                print(f'Player of type {type(player)} tried to flip a flipped card. They are disqualified.')
                self.kill_player(player)
        player.roles[flipped_role_idx] = Role.NONE
        if all(role == Role.NONE for role in player.roles):
            self.kill_player(player)

        # Notify other players
        obs = Observation(ObservationType.FLIP_ROLE, FlipCard(flipped_role), idx, idx)
        self.send_observation(obs)

    def is_alive(self, player: Player):
        if player not in self.players:
            raise ValueError('Player not in game passed to is_alive()')
        return self.alive_players[player.idx]

    def kill_player(self, player: Player):
        if player not in self.players:
            raise ValueError('Player not in game passed to kill_player()')
        self.alive_players[player.idx] = False
        if self.verbosity > 1:
            print(f'Player {player.idx} has been killed')

    def get_legal_moves(self, player: Player) -> List[Tuple[Move, int]]:
        cp_idx = player.idx
        alive_indices = [i for i in range(len(self.alive_players)) if i != cp_idx and self.alive_players[i]]

        if player.coins >= 10:
            return [(Move.COUP, player_against) for player_against in alive_indices]

        legal_moves = [(move, cp_idx) for move in [Move.INCOME, Move.FOREIGN_AID, Move.SWAP_CARDS, Move.TAX]]
        if player.coins >= 3:
            legal_moves += [(Move.ASSASSINATE, player_against) for player_against in alive_indices]
        if player.coins >= 7:
            legal_moves += [(Move.COUP, player_against) for player_against in alive_indices]
        for player_against in alive_indices:
            if self.players[player_against].coins > 0:
                legal_moves.append((Move.STEAL, player_against))

        return legal_moves

    def get_legal_responses(self, attacked: bool, move: Move):
        legal_responses = [Response.NOTHING]
        if move in BLOCKABLE_MOVES_ALL or (attacked and move in BLOCKABLE_MOVES):
            legal_responses.append(Response.BLOCK)
        if move in BS_ABLE_MOVES:
            legal_responses.append(Response.CALL_BS)
        return legal_responses

    def send_observation(self, obs: Observation):
        visible_state = VisibleState(
            [player.coins for player in self.players],
            [len([i for i in player.roles if i != Role.NONE]) for player in self.players])
        obs.visible_state = visible_state
        for player in self.players:
            player.get_observation(obs)
        if self.verbosity > 1:
            print(obs)

    def get_game_state(self):
        return {'verbosity': 0, 'cur_turn': self.cur_turn, 'alive_players': self.alive_players,
                'roles_per_player': self.roles_per_player,
                'player_types': self.player_types, 'middle_cards': self.middle_cards,
                'players': self.players}

    def print_game_state(self):
        for i, player in enumerate(self.players):
            print(f'Player {i}: {player.coins} coins and roles {", ".join([role.name for role in player.roles])}')
        print(f'Deck: {", ".join([role.name for role in self.middle_cards])}\n')


if __name__ == '__main__':
    from q import QPlayer
    from mcts import MCTSPlayer
    import matplotlib.pyplot as plt

    # Q = defaultdict(defaultdict(int).copy)  # action value estimates
    # N = defaultdict(defaultdict(int).copy)  # visit counts
    Q = pickle.load(open('q_weights', 'rb'))
    N = pickle.load(open('n_weights', 'rb'))

    winners = np.array([0, 0, 0, 0])
    last = np.copy(winners)
    rates = []
    for i in range(0):
        sim = Simulator.from_start([RandomPlayer, RandomPlayer, RandomPlayer, QPlayer],
                                   params=[{}, {}, {},
                                           {'Q': Q, 'N': N,
                                            'c': .01,
                                            'depth': 100,
                                            'num_simulations': 10,
                                            'alpha': 0.1}],
                                   verbosity=0)
        winner = sim.run_game()
        winners[winner] += 1
        if i % 500 == 0:
            print(i)
            print(winners)
            print(winners - last)
            print((winners[3] / np.sum(winners)) * 100)
            print(((winners - last)[3] / np.sum(winners - last)) * 100)
            rates.append(((winners - last)[3] / np.sum(winners - last)) * 100)
            last = np.copy(winners)
    plt.plot(rates[1:])
    plt.show()
    print()
    # pickle.dump(Q, open('q_weights', 'wb'))
    # pickle.dump(N, open('n_weights', 'wb'))

    for i in range(20):
        # sim = Simulator.from_start([RandomPlayer, RandomPlayer, RandomPlayer, HeuristicPlayer], params=[{}, {}, {}, {}], verbosity=0)
        sim = Simulator.from_start([RandomPlayer, RandomPlayer, RandomPlayer, MCTSPlayer],
                                   params=[{}, {}, {},
                                           {'Q': Q, 'N': N,
                                            'c': .01,
                                            'depth': 100,
                                            'num_simulations': 10,
                                            'alpha': 0.1}],
                                   verbosity=0)
        # sim = Simulator.from_start([RandomPlayer, RandomPlayer, RandomPlayer, QPlayer], params=[{}, {}, {}, {'Q':Q, 'N':N, 'c':.01, 'alpha':0.1}], verbosity=0)
        winner = sim.run_game()
        winners[winner] += 1
        if i % 1 == 0:
            print(i)
            print(winners)
            print(winners - last)
            print((winners[3] / np.sum(winners)) * 100)
            print(((winners - last)[3] / np.sum(winners - last)) * 100)
            rates.append(((winners - last)[3] / np.sum(winners - last)) * 100)
            last = np.copy(winners)
    plt.plot(rates[1:])
    plt.show()
    print()
