import datetime
import math
import os

import numpy
import torch

from .abstract_game import AbstractGame
import chess
from tqdm import tqdm
import random

class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = (3, 8, 8)  # Dimensions of the game observation, must be 3 (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(64 * 64))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "random"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 4 # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = True
        self.max_moves = 400  # Maximum number of moves if game is not finished before
        self.num_simulations = 400  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 3  # Number of blocks in the ResNet
        self.channels = 64  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        self.resnet_fc_reward_layers = [64]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [64]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [64]  # Define the hidden layers in the policy head of the prediction network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network



        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 1000000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.003  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000



        ### Replay Buffer
        self.replay_buffer_size = 3000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 121  # Number of game moves to keep for every batch element
        self.td_steps = 121  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = False  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it



    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Chess()

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def close(self):
        """
        Properly close the game.
        """
        pass

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        valid = False
        while not valid:
            valid, action = self.env.human_input_to_action()
        return action

    def action_to_string(self, action):
        """
        Convert an action number to a string representing the action.
        Args:
            action_number: an integer from the action space.
        Returns:
            String representing the action.
        """
        return self.env.action_to_human_input(action)

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_agent()

    def play_mode(self, mode, games, verbose):
        return self.env.play_mode(mode, games, verbose)

class Chess:

    def __init__(self, elo=1000):
        self._board = chess.Board()

    def to_play(self):
        # return 0 if self.player == 1 else 1
        return 0 if self._board.turn else 1

    def reset(self):
        self._board.reset()
        return self._observation()

    def step(self, action):
        move = self._to_move(action)
        if move not in self._board.legal_moves:
            move = chess.Move.from_uci(move.uci()+'q') # assume promotion to queen
            if move not in self._board.legal_moves:
                raise ValueError(
                    f"Illegal move {action} for board position {self._board.fen()}"
                )
        self._board.push(move)
        observation = self._observation()
        reward = self._reward()
        done = self._board.is_game_over()
        return observation, reward*20, done

    def _reward(self):
        outcome = self._board.outcome()
        if outcome is not None:
            if outcome.winner is None:
                # draw return very little value
                print(f"Draw")
                return 0
            else:
                print(f"{outcome.winner} Win")
                return 1 if outcome.winner else -1
        return 0


    def _observation(self):
        board_player1 = numpy.zeros((8,8))
        board_player2 = numpy.zeros((8,8))
        for sq,pc in self._board.piece_map().items():
            if pc.color:
                board_player1[int(sq/8)][int(sq%8)] = (pc.piece_type)
            else:
                board_player2[int(sq/8)][int(sq%8)] = (pc.piece_type)
        # board_player1 = numpy.where(a > 0, a, 0)
        # board_player2 = numpy.where(a < 0, -a, 0)
        if self._board.turn:
            board_to_play = numpy.full((8,8), 1)
        else:
            board_to_play = numpy.full((8,8), -1)
        return numpy.array([board_player1, board_player2, board_to_play], dtype="int32")


    def legal_actions(self):
        return [self._from_move(move) for move in self._board.legal_moves]

    def render(self):
        print(self._board)

    def _move_from_uci(self, uci):
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            print('expected an UCI move')
            return None
        if move not in self._board.legal_moves:
            print('expected a valid move')
            return None
        return move

    def _from_move(self, move):
      return move.from_square*64+move.to_square

    def _to_move(self, action):
      to_sq = action % 64
      from_sq = int(action / 64)
      return chess.Move(from_sq, to_sq)

    def human_input_to_action(self):
        moves = [self._board.uci(move) for move in self._board.legal_moves]
        print("Legal action: " + (",".join(moves)))
        human_input = input("Enter an action: ")
        move = self._move_from_uci(human_input)
        if move:
            return True, self._from_move(move)
        return False, -1

    def action_to_human_input(self, action):
        move = self._to_move(action)
        return self._board.uci(move)

    def random_player(self):
        move = random.choice(self.legal_actions())
        return move

    def expert_agent(self):
        from stockfish import Stockfish
        self._stockfish = Stockfish(parameters={"Threads": 2, "Minimum Thinking Time": 30})
        self._stockfish.set_elo_rating(1000)
        self._stockfish.set_fen_position(self._board.fen())
        uci_move = self._stockfish.get_best_move()
        move = self._move_from_uci(uci_move.strip())
        return self._from_move(move)

    def play_mode(self, mode, games, verbose=False):
        games = int(games / 2)
        oneWon = 0
        twoWon = 0
        draws = 0

        if mode == 1:
            p1 = self.random_player
            p2 = self.expert_agent
        else:
            p1 = self.random_player
            p2 = self.random_player

        rotation = [p1, p2]
        for _ in tqdm(range(games), desc="Arena.playGames (1)"):
            done =  False
            self.reset()
            while not done:
                for play in rotation:
                    action = play()
                    observation, reward, done = self.step(action)
                    if verbose:
                        self.render()
                    if done:
                        if reward ==1:
                            oneWon += 1
                        elif reward == -1:
                            twoWon += 1
                        else:
                            draws += 1
                        break

        rotation = [p2, p1]
        for _ in tqdm(range(games), desc="Arena.playGames (2)"):
            done =  False
            self.reset()
            while not done:
                for play in rotation:
                    action = play()
                    observation, reward, done = self.step(action)
                    if verbose:
                        self.render()
                    if done:
                        if reward ==1:
                            twoWon += 1
                        elif reward == -1:
                            oneWon += 1
                        else:
                            draws += 1
                        break
        print(f"Player 1: {oneWon}, Player 2: {twoWon}, Draws: {draws})")
