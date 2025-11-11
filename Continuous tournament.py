#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import subprocess
import glob


# In[2]:


# Connect 4 Self-Play Arena
# Two Q-Networks battle each other for continuous improvement

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from board_processor import BoardProcessor
from feature_generator import FeatureGenerator
from typing import Optional, List, Dict, Tuple
import pickle
from dataclasses import dataclass
import time


# In[24]:


@dataclass
class SelfPlayConfig:
    """Configuration for self-play sessions"""
    model_path_alpha: str = "qnet_mc_pretrained.pth"
    model_path_bravo: str = "qnet_mc_pretrained.pth"
    epsilon_alpha: float = 0.1
    epsilon_bravo: float = 0.1
    alpha_plays_first: bool = True
    num_games: int = 100
    verbose: bool = True
    save_games: bool = False
    game_save_path: str = "selfplay_games.pkl"


# In[4]:


@dataclass
class GameResult:
    """Store results of a single game"""
    moves: List[int]
    winner: int  # 1 for alpha, -1 for bravo, 0 for draw
    game_length: int
    game_code: str
    alpha_first: bool
    epsilon_alpha: float
    epsilon_bravo: float


# In[5]:


class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_sizes=(256, 128, 64, 32, 16, 8)):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.Tanh())
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# In[6]:


class SelfPlayArena:
    """Manages self-play between two Q-Networks"""

    def __init__(self, config: SelfPlayConfig):
        self.config = config
        self.feature_gen = FeatureGenerator()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Calculate feature dimensions
        _, dummy_features = self.feature_gen.convolution_feature_gen([[] for _ in range(7)])
        self.feature_dim = len(dummy_features) * 2  # State-action pairs

        # Load models and scalers
        self.model_alpha, self.scaler_alpha = self._load_model(config.model_path_alpha)
        self.model_bravo, self.scaler_bravo = self._load_model(config.model_path_bravo)

        # Game statistics
        self.reset_stats()

        print(f"Arena initialized! Using {self.device}")
        print(f"Feature dimension: {self.feature_dim}")
        print(f"Alpha model: {config.model_path_alpha}")
        print(f"Bravo model: {config.model_path_bravo}")

    def _load_model(self, model_path: str) -> Tuple[QNetwork, object]:
        """Load a model and its scaler"""
        model = QNetwork(input_dim=self.feature_dim).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        scaler = checkpoint['scaler']
        return model, scaler

    def reset_stats(self):
        """Reset game statistics"""
        self.stats = {
            'alpha_wins': 0,
            'bravo_wins': 0,
            'draws': 0,
            'total_games': 0,
            'avg_game_length': 0,
            'game_results': []
        }

    def get_q_value(self, state_list: List[List[int]], move: int, player: int,
                   model: QNetwork, scaler) -> float:
        """Get Q-value for a state-action pair"""
        _, curr_feats = self.feature_gen.convolution_feature_gen(state_list)
        next_state = [col[:] for col in state_list]
        next_state[move].append(player)
        _, next_feats = self.feature_gen.convolution_feature_gen(next_state)

        features = np.concatenate([curr_feats, next_feats])
        scaled = scaler.transform([features])

        with torch.no_grad():
            return model(torch.FloatTensor(scaled).to(self.device)).item()

    def get_ai_move(self, state_list: List[List[int]], player: int, epsilon: float,
                   model: QNetwork, scaler) -> Tuple[int, float, Dict[int, float]]:
        """Select AI move using epsilon-greedy strategy"""
        valid = [c for c in range(7) if len(state_list[c]) < 6]
        q_values = {}

        # Calculate Q-values for all valid moves
        for col in valid:
            q = self.get_q_value(state_list, col, player, model, scaler) * player
            q_values[col] = q

        # Epsilon-greedy selection
        if np.random.random() < epsilon:
            selected_col = int(np.random.choice(valid))
        else:
            selected_col = max(q_values.keys(), key=lambda k: q_values[k])

        return selected_col, q_values[selected_col], q_values

    def check_win(self, state_list: List[List[int]]) -> int:
        """Check for win using convolution features. Returns 1, -1, or 0"""
        _, features = self.feature_gen.convolution_feature_gen(state_list)
        if 4 in features:
            return 1
        elif -4 in features:
            return -1
        return 0

    def play_single_game(self, game_num: int = 0) -> GameResult:
        """Play a single game between Alpha and Bravo"""
        board = BoardProcessor()
        moves = []

        # Determine who plays first and assign player values
        if self.config.alpha_plays_first:
            alpha_player, bravo_player = 1, -1
            current_is_alpha = True
        else:
            alpha_player, bravo_player = -1, 1
            current_is_alpha = False

        if self.config.verbose:
            starter = "Alpha" if current_is_alpha else "Bravo"
            print(f"\nGame {game_num + 1}: {starter} plays first")

        # Game loop
        while True:
            # Determine current player and model
            if current_is_alpha:
                player_value = alpha_player
                model, scaler = self.model_alpha, self.scaler_alpha
                epsilon = self.config.epsilon_alpha
                player_name = "Alpha"
            else:
                player_value = bravo_player
                model, scaler = self.model_bravo, self.scaler_bravo
                epsilon = self.config.epsilon_bravo
                player_name = "Bravo"

            # Get move
            col, q_val, q_values = self.get_ai_move(
                board.state_list, player_value, epsilon, model, scaler
            )

            moves.append(col)
            board.generate_state_list(moves)

            if self.config.verbose:
                print(f"{player_name} plays column {col} (Q={q_val:.3f})")

            # Check for game end
            winner = self.check_win(board.state_list)
            if winner != 0:
                # Convert winner to Alpha/Bravo perspective
                if winner == alpha_player:
                    result_winner = 1  # Alpha wins
                    winner_name = "Alpha"
                else:
                    result_winner = -1  # Bravo wins
                    winner_name = "Bravo"

                if self.config.verbose:
                    print(f"{winner_name} wins in {len(moves)} moves! Code: {board.moves_code()}")
                break

            if len(moves) >= 42:
                result_winner = 0
                if self.config.verbose:
                    print(f"Draw in {len(moves)} moves! Code: {board.moves_code()}")
                break

            # Switch players
            current_is_alpha = not current_is_alpha

        return GameResult(
            moves=moves,
            winner=result_winner,
            game_length=len(moves),
            game_code=board.moves_code(),
            alpha_first=self.config.alpha_plays_first,
            epsilon_alpha=self.config.epsilon_alpha,
            epsilon_bravo=self.config.epsilon_bravo
        )

    def run_tournament(self) -> Dict:
        """Run a tournament of multiple games"""
        print(f"\n=== Starting tournament: {self.config.num_games} games ===")
        print(f"Alpha eps={self.config.epsilon_alpha}, Bravo eps={self.config.epsilon_bravo}")

        start_time = time.time()

        for game_num in range(self.config.num_games):
            result = self.play_single_game(game_num)

            # Update statistics
            if result.winner == 1:
                self.stats['alpha_wins'] += 1
            elif result.winner == -1:
                self.stats['bravo_wins'] += 1
            else:
                self.stats['draws'] += 1

            self.stats['total_games'] += 1
            self.stats['game_results'].append(result)

            # Alternate who plays first (optional)
            if (game_num + 1) % 2 == 0:
                self.config.alpha_plays_first = not self.config.alpha_plays_first

        # Calculate final statistics
        total_length = sum(r.game_length for r in self.stats['game_results'])
        self.stats['avg_game_length'] = total_length / self.config.num_games
        elapsed = time.time() - start_time

        # Print summary
        self._print_tournament_summary(elapsed)

        # Find and display most common game
        self._display_most_common_game()

        # Save results if requested
        if self.config.save_games:
            self._save_results()

        return self.stats

    def _print_tournament_summary(self, elapsed_time: float):
        """Print tournament results"""
        print(f"\n=== Tournament Results ===")
        print(f"Games played: {self.stats['total_games']}")
        print(f"Alpha wins: {self.stats['alpha_wins']} ({self.stats['alpha_wins']/self.stats['total_games']*100:.1f}%)")
        print(f"Bravo wins: {self.stats['bravo_wins']} ({self.stats['bravo_wins']/self.stats['total_games']*100:.1f}%)")
        print(f"Draws: {self.stats['draws']} ({self.stats['draws']/self.stats['total_games']*100:.1f}%)")
        print(f"Average game length: {self.stats['avg_game_length']:.1f} moves")
        print(f"Time elapsed: {elapsed_time:.1f} seconds")
        print(f"Games per second: {self.stats['total_games']/elapsed_time:.1f}")

    def _display_most_common_game(self):
        """Find and display the most common game pattern"""
        if not self.stats['game_results']:
            print("No games to analyze!")
            return

        # Count game codes
        from collections import Counter
        game_codes = [result.game_code for result in self.stats['game_results']]
        code_counts = Counter(game_codes)

        if not code_counts:
            print("No game codes found!")
            return

        # Find most common
        most_common_code, count = code_counts.most_common(1)[0]

        print(f"\n=== MOST COMMON GAME PATTERN ===")
        print(f"Game code: {most_common_code}")
        print(f"Occurred {count} times out of {len(game_codes)} games ({count/len(game_codes)*100:.1f}%)")

        # Recreate and display the game
        try:
            board = BoardProcessor()
            moves = board.decode_moves_code(most_common_code)
            board.generate_state_list(moves)

            print(f"Move sequence: {moves}")
            print(f"Game length: {len(moves)} moves")
            print(f"Final board position:")
            board.display_board()

            # Check winner
            winner = self.check_win(board.state_list)
            if winner == 1:
                print("Winner: Player 1 (X)")
            elif winner == -1:
                print("Winner: Player -1 (O)")
            else:
                print("Result: Draw")

        except Exception as e:
            print(f"Error decoding game: {e}")

    def _save_results(self):
        """Save tournament results to file"""
        save_data = {
            'config': self.config,
            'stats': self.stats,
            'timestamp': time.time()
        }

        with open(self.config.game_save_path, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"Results saved to {self.config.game_save_path}")

    def plot_results(self):
        """Plot tournament results"""
        if not self.stats['game_results']:
            print("No games played yet!")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # Win percentages pie chart
        labels = ['Alpha Wins', 'Bravo Wins', 'Draws']
        sizes = [self.stats['alpha_wins'], self.stats['bravo_wins'], self.stats['draws']]
        colors = ['lightblue', 'lightcoral', 'lightgray']
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Win Distribution')

        # Game length histogram
        lengths = [r.game_length for r in self.stats['game_results']]
        ax2.hist(lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Game Length (moves)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Game Length Distribution')
        ax2.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.1f}')
        ax2.legend()

        # Running win rate
        alpha_wins = []
        running_alpha = 0
        for i, result in enumerate(self.stats['game_results']):
            if result.winner == 1:
                running_alpha += 1
            alpha_wins.append(running_alpha / (i + 1))

        ax3.plot(alpha_wins, label='Alpha Win Rate', color='blue')
        ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Game Number')
        ax3.set_ylabel('Win Rate')
        ax3.set_title('Running Win Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Win rate by starting player
        alpha_first_wins = sum(1 for r in self.stats['game_results']
                              if r.alpha_first and r.winner == 1)
        alpha_first_total = sum(1 for r in self.stats['game_results'] if r.alpha_first)

        bravo_first_wins = sum(1 for r in self.stats['game_results']
                              if not r.alpha_first and r.winner == -1)
        bravo_first_total = sum(1 for r in self.stats['game_results'] if not r.alpha_first)

        if alpha_first_total > 0 and bravo_first_total > 0:
            first_player_advantage = [
                alpha_first_wins / alpha_first_total,
                bravo_first_wins / bravo_first_total
            ]
            ax4.bar(['Alpha plays first', 'Bravo plays first'], first_player_advantage,
                   color=['lightblue', 'lightcoral'])
            ax4.set_ylabel('Win Rate when playing first')
            ax4.set_title('First Player Advantage')
            ax4.set_ylim(0, 1)
        else:
            ax4.text(0.5, 0.5, 'Not enough alternating games', ha='center', va='center')
            ax4.set_title('First Player Advantage (insufficient data)')

        plt.tight_layout()
        plt.show()


# In[19]:


import time
from datetime import timedelta
from trueskill import Rating, rate_1vs1
import random

# In[10]:


# Download all .pth files
subprocess.run([
    'scp',
    'az:work/connect4-qlearning/*.pth',
    'models_pth/'
], check=True)

print(f"Downloaded models to: {os.path.abspath('models_pth')}")


# Configuration
GAMES_PER_MATCH = 24  # Reduced from 500 for faster iterations
KAPPA = 3  # Conservative ranking multiplier
MAX_ROUNDS = 150
SEPARATION_THRESHOLD = 1.5  # mu - 3*sigma difference needed
EPSILON = 0.1

# Initialize ratings
model_paths = glob.glob('models_pth/*.pth')
random.shuffle(model_paths)
# model_paths = model_paths[:15]
ratings = {path: Rating() for path in model_paths}
match_history = {}  # Track who played whom



# In[20]:


def conservative_score(rating):
    return rating.mu - KAPPA * rating.sigma

def make_swiss_pairs(ranked_models, match_history):
    """Pair adjacent ranks, avoiding recent rematches"""
    pairs = []
    used = set()
    
    for i in range(0, len(ranked_models)-1, 2):
        if i >= len(ranked_models) - 1:
            break
        
        a, b = ranked_models[i], ranked_models[i+1]
        
        # Skip if recently played
        pair_key = tuple(sorted([a, b]))
        if pair_key in match_history and match_history[pair_key] >= 3:
            # Try next opponent
            if i+2 < len(ranked_models):
                b = ranked_models[i+2]
                pair_key = tuple(sorted([a, b]))
        
        if a not in used and b not in used:
            pairs.append((a, b))
            used.add(a)
            used.add(b)
            match_history[pair_key] = match_history.get(pair_key, 0) + 1
    
    return pairs


# In[21]:


def play_match(model_a, model_b, n_games, epsilon):
    """Play n games and return win counts"""
    config = SelfPlayConfig(
        model_path_alpha=model_a,
        model_path_bravo=model_b,
        epsilon_alpha=epsilon,
        epsilon_bravo=epsilon,
        num_games=n_games,
        verbose=False
    )
    arena = SelfPlayArena(config)
    results = arena.run_tournament()
    return results['alpha_wins'], results['bravo_wins'], results['draws']



# In[22]:


# Main loop
start_time = time.time()
round_num = 0

print(f"Starting adaptive tournament with {len(model_paths)} models")
print(f"Target: Find stable top 3\n")

while round_num < MAX_ROUNDS:
    round_num += 1
    round_start = time.time()
    
    # 1. Rank by conservative score
    ranked = sorted(model_paths, key=lambda p: conservative_score(ratings[p]), reverse=True)
    
    # 2. Make pairs (Swiss pairing)
    pairs = make_swiss_pairs(ranked, match_history)
    
    # 3. Play matches and update ratings
    print(f"\n{'='*60}")
    print(f"ROUND {round_num}: {len(pairs)} matches")
    print(f"{'='*60}")
    
    for idx, (model_a, model_b) in enumerate(pairs, 1):
        a_name = os.path.basename(model_a)
        b_name = os.path.basename(model_b)
        
        print(f"[{idx}/{len(pairs)}] {a_name} vs {b_name}")
        
        a_wins, b_wins, draws = play_match(model_a, model_b, GAMES_PER_MATCH, EPSILON)
        
        # Create shuffled sequence of actual game outcomes
        game_results = ['A'] * a_wins + ['B'] * b_wins + ['D'] * draws
        random.shuffle(game_results)
        
        # Update TrueSkill game-by-game in random order
        for result in game_results:
            if result == 'A':
                ratings[model_a], ratings[model_b] = rate_1vs1(ratings[model_a], ratings[model_b])
            elif result == 'B':
                ratings[model_b], ratings[model_a] = rate_1vs1(ratings[model_b], ratings[model_a])
            else:  # Draw
                ratings[model_a], ratings[model_b] = rate_1vs1(ratings[model_a], ratings[model_b], drawn=True)
        print(f"  Result: {a_wins}-{b_wins}-{draws}")
        print(f"  Ratings: {ratings[model_a].mu:.1f}+-{ratings[model_a].sigma:.1f} vs {ratings[model_b].mu:.1f}+-{ratings[model_b].sigma:.1f}")

    # 4. Check convergence
    cons_scores = sorted([conservative_score(ratings[p]) for p in model_paths], reverse=True)
    top3_min = cons_scores[2]
    rest_max = max(cons_scores[3:]) if len(cons_scores) > 3 else -999
    separation = top3_min - rest_max
    
    round_time = time.time() - round_start
    elapsed = time.time() - start_time
    
    print(f"\nRound stats:")
    print(f"  Time: {timedelta(seconds=int(round_time))}")
    print(f"  Top 3 separation: {separation:.3f}")
    print(f"  Total elapsed: {timedelta(seconds=int(elapsed))} ({elapsed/3600:.2f}h)")
    
    # Show current top 5
    print(f"\n  Current rankings:")
    for i, path in enumerate(ranked[:5], 1):
        r = ratings[path]
        print(f"    {i}. {os.path.basename(path)}: {r.mu:.1f}+-{r.sigma:.1f} (cons: {conservative_score(r):.1f})")
    
    if separation > SEPARATION_THRESHOLD and all(ratings[p].sigma < 2.0 for p in ranked[:3]):
        print(f"\n✓ Top 3 converged with separation {separation:.3f}")
        break


# In[23]:


# Final results
print(f"\n{'='*60}")
print(f"TOURNAMENT COMPLETE")
print(f"{'='*60}")
print(f"Rounds: {round_num}/({MAX_ROUNDS} is the maximum)")
print(f"Total matches: {sum(match_history.values())}")
print(f"Total games: {sum(match_history.values()) * GAMES_PER_MATCH}")
print(f"Total time: {timedelta(seconds=int(time.time() - start_time))} ({(time.time()-start_time)/3600:.2f}h)")
print(f"\nTOP 3 MODELS:")

final_ranking = sorted(model_paths, key=lambda p: conservative_score(ratings[p]), reverse=True)
for i, path in enumerate(final_ranking, 1):
    r = ratings[path]
    print(f"{i}. {os.path.basename(path)}")
    print(f"   mu={r.mu:.2f}, sigma={r.sigma:.2f}, conservative={conservative_score(r):.2f}")


# In[ ]:




