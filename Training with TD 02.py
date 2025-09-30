#%%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
# Named tuple for clarity
# noinspection PyTypeChecker
TrainingTuple = namedtuple('TrainingTuple', ['state_action', 'initial_q', 'target_q', 'td_error', 'move_number', 'game_length'])
#%%
from board_processor import BoardProcessor
from feature_generator import FeatureGenerator
import os

class QNetwork(nn.Module):
    def __init__(self, input_dim=138):
        super().__init__()
        layers = []
        for h in [256, 128, 64, 32, 16, 8]:
            layers.extend([nn.Linear(input_dim, h), nn.Tanh()])
            input_dim = h
        layers.extend([nn.Linear(h, 1), nn.Tanh()])
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)
#%%

def get_current_q(state_action_features, online_model, scaler, player):
    """Get current Q-value prediction from online model"""
    scaled = scaler.transform([state_action_features])
    with torch.no_grad():
        q = online_model(torch.FloatTensor(scaled).to(online_model.device if hasattr(online_model, 'device') else 'cpu')).item()
    return q * player  # Adjust for player perspective

def calculate_target_q(moves, position_i, player, online_model, target_model, scaler, feature_gen, gamma=0.99):
    """Calculate target Q-value using Double DQN logic"""
    # Check if game ends after our move (position i+1)
    board_after_our_move = BoardProcessor()
    board_after_our_move.generate_state_list(moves[:position_i+1])
    _, feats_after_our_move = feature_gen.convolution_feature_gen(board_after_our_move.state_list)

    # Terminal after our move?
    if 4 in feats_after_our_move:
        return 1 * player  # We win
    elif -4 in feats_after_our_move:
        return -1 * player  # We lose (shouldn't happen)
    elif position_i + 1 >= len(moves):
        return 0  # Draw

    # Check if game ends after opponent's move (position i+2)
    if position_i + 3 >= 42:
        # print("^"*15 + f"Debug - IT GOT triggered when {position_i} plus two is greater than 42")
        return 0  # Draw
    # else: print("^"*15 + f"Debug - ever gets triggered when {position_i} plus two is greater than 42")

    board_after_opp = BoardProcessor()
    board_after_opp.generate_state_list(moves[:position_i+2])
    _, feats_after_opp = feature_gen.convolution_feature_gen(board_after_opp.state_list)

    if 4 in feats_after_opp or -4 in feats_after_opp:
        return -1   # Opponent wins

    # Non-terminal: calculate Q-value of next state
    next_board = BoardProcessor()
    next_board.generate_state_list(moves[:position_i+2])
    _, next_curr_feats = feature_gen.convolution_feature_gen(next_board.state_list)

    # Get Q-values for all possible next moves using ONLINE network for selection
    online_q_values = []
    for col in range(7):
        if len(next_board.state_list[col]) < 6:  # Legal move
            next_state = [c[:] for c in next_board.state_list]
            next_state[col].append(player)  # Same player's turn
            _, next_feats = feature_gen.convolution_feature_gen(next_state)

            # Check for immediate win
            if 4 * player in next_feats:
                online_q_values.append((col, 1.0))
            else:
                # Get Q-value from ONLINE model
                features = np.concatenate([next_curr_feats, next_feats])
                scaled = scaler.transform([features])
                with torch.no_grad():
                    q = online_model(torch.FloatTensor(scaled).to(online_model.device if hasattr(online_model, 'device') else 'cpu')).item() * player
                    online_q_values.append((col, q))

    if not online_q_values:
        return 0  # No legal moves = draw

    # DOUBLE DQN: Online network selects best action
    best_action = max(online_q_values, key=lambda x: x[1])[0]

    # TARGET network evaluates the selected action
    best_next_state = [c[:] for c in next_board.state_list]
    best_next_state[best_action].append(player)
    _, best_next_feats = feature_gen.convolution_feature_gen(best_next_state)
    target_q_value = 0.0
    # Check for immediate win with selected action
    if 4 * player in best_next_feats:
        target_q_value = 1.0
    else:
        # Evaluate using TARGET network
        best_features = np.concatenate([next_curr_feats, best_next_feats])
        best_scaled = scaler.transform([best_features])
        with torch.no_grad():
            target_q_value = target_model(torch.FloatTensor(best_scaled).to(target_model.device if hasattr(target_model, 'device') else 'cpu')).item() * player

    return gamma * target_q_value
#%%
#Most advanced so far
def generate_training_tuples_with_td(game_codes, online_model, target_model, scaler,
                                     feature_gen,
                                     alpha=0.1, gamma=0.99, max_tuples=None):
    """
    Generate (state_action_features, target_q, td_error, metadata) tuples from game codes
    TD error is calculated during generation to avoid redundant forward passes
    """
    training_tuples = []

    for game_idx, game_code in enumerate(game_codes):
        board = BoardProcessor()
        moves = board.decode_moves_code(game_code)
        board.generate_state_list(moves)
        game_length = len(moves)

        # Process each non-terminal position
        for i in range(len(moves) - 1):  # Skip final position
            # Current state and player
            temp_board = BoardProcessor()
            temp_board.generate_state_list(moves[:i])
            player = 1 if (i % 2) == 0 else -1

            # Get current state features
            _, curr_feats = feature_gen.convolution_feature_gen(temp_board.state_list)

            # Action taken and resulting state
            action = moves[i]
            next_state = [col[:] for col in temp_board.state_list]
            next_state[action].append(player)
            _, next_feats = feature_gen.convolution_feature_gen(next_state)

            # Create state-action input features
            state_action_features = np.concatenate([curr_feats, next_feats])

            # Get CURRENT Q-value (we're computing this anyway!)
            current_q = get_current_q(state_action_features, online_model, scaler, player)

            # Calculate target Q-value using Double DQN
            target_q_raw = calculate_target_q(moves, i, player, online_model, target_model,
                                             scaler, feature_gen, gamma)

            # TD error (before applying alpha)
            td_error = target_q_raw - current_q

            # New Q value after TD update
            new_q = current_q + alpha * td_error

            # Store tuple with all information
            tuple_data = TrainingTuple(
                state_action=state_action_features,
                initial_q=current_q * player,
                target_q=new_q * player,
                td_error=abs(td_error),  # Store absolute TD error for prioritization
                move_number=i,
                game_length=game_length
            )
            training_tuples.append(tuple_data)

            # Early exit if we have enough tuples
            if max_tuples and len(training_tuples) >= max_tuples:
                return training_tuples

    return training_tuples
#%%

# This one is smart - can prioritize samples into batches
class SmartReplayBuffer:
    """
    Replay buffer that uses pre-computed TD errors and game metadata
    """
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, tuple_data):
        """Add a training tuple to the buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(tuple_data)
        else:
            self.buffer[self.position] = tuple_data
        self.position = (self.position + 1) % self.capacity

    def sample_prioritized(self, batch_size, alpha=0.6, beta=0.4,
                          endgame_bonus=2.0, endgame_threshold=35):
        """
        Sample using pre-computed TD errors with endgame position bonus

        Args:
            alpha: Priority exponent for TD errors
            beta: Importance sampling correction
            endgame_bonus: Multiplier for endgame positions
            endgame_threshold: Positions after this move get bonus
        """
        if len(self.buffer) < batch_size:
            return None

        # Calculate priorities using stored TD errors
        priorities = []
        for tuple_data in self.buffer:
            priority = (tuple_data.td_error + 1e-6) ** alpha

            # Bonus for endgame positions
            if tuple_data.move_number >= endgame_threshold:
                priority *= endgame_bonus

            # Additional bonus for very close games
            if tuple_data.game_length >= 40:  # Near-draw games
                priority *= 1.5

            priorities.append(priority)

        # Convert to probabilities
        priorities = np.array(priorities)
        probs = priorities / priorities.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        # Create batch
        batch = [self.buffer[idx] for idx in indices]

        # states = torch.FloatTensor([t.state_action for t in batch])
        states = torch.FloatTensor(np.array([t.state_action for t in batch]))
        targets = torch.FloatTensor([t.target_q for t in batch])
        weights = torch.FloatTensor(weights)
        td_errors = torch.FloatTensor([t.td_error for t in batch])

        return states, targets, weights, td_errors, indices

    def sample_uniform(self, batch_size):
        """Simple uniform sampling for comparison"""
        if len(self.buffer) < batch_size:
            return None

        indices = np.random.choice(len(self.buffer), batch_size)
        batch = [self.buffer[idx] for idx in indices]

        # states = torch.FloatTensor([t.state_action for t in batch])
        states = torch.FloatTensor(np.array([t.state_action for t in batch]))
        targets = torch.FloatTensor([t.target_q for t in batch])

        return states, targets

    def plot_td_error_analysis(self, figsize=(15, 5), alpha=0.6, s=20):
        """
        Plot TD error analysis with three scatter plots:
        1. TD Error vs Move Number
        2. TD Error vs Game Length
        3. TD Error vs Endgame Ratio

        Works with existing TrainingTuple: ['state_action', 'target_q', 'td_error', 'move_number', 'game_length']
        """


        if not self.buffer:
            print("Buffer is empty! Add some training examples first.")
            return

        # Extract data from TrainingTuples
        td_errors = [t.td_error for t in self.buffer]
        move_numbers = [t.move_number for t in self.buffer]
        game_lengths = [t.game_length for t in self.buffer]
        # Calculate endgame ratio from existing fields
        endgame_ratios = [move_num / max(game_len - 1, 1) for move_num, game_len in zip(move_numbers, game_lengths)]

        # Create three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

        # Plot 1: TD Error vs Move Number
        ax1.scatter(move_numbers, td_errors, alpha=alpha, s=s, color='blue')
        ax1.set_xlabel('Move Number')
        ax1.set_ylabel('TD Error')
        ax1.set_title('TD Error vs Move Number')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Plot 2: TD Error vs Game Length
        ax2.scatter(game_lengths, td_errors, alpha=alpha, s=s, color='green')
        ax2.set_xlabel('Game Length (total moves)')
        ax2.set_ylabel('TD Error')
        ax2.set_title('TD Error vs Game Length')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Plot 3: TD Error vs Endgame Ratio
        ax3.scatter(endgame_ratios, td_errors, alpha=alpha, s=s, color='red')
        ax3.set_xlabel('Endgame Ratio (move/total_moves)')
        ax3.set_ylabel('TD Error')
        ax3.set_title('TD Error vs Endgame Ratio')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

        # Print some statistics
        print(f"\nTD Error Analysis Statistics:")
        print(f"Total examples: {len(self.buffer)}")
        print(f"TD Error - Mean: {np.mean(td_errors):.4f}, Std: {np.std(td_errors):.4f}")
        print(f"TD Error - Min: {np.min(td_errors):.4f}, Max: {np.max(td_errors):.4f}")
        print(f"Move numbers - Range: {np.min(move_numbers)} to {np.max(move_numbers)}")
        print(f"Game lengths - Range: {np.min(game_lengths)} to {np.max(game_lengths)}")
        print(f"Endgame ratios - Range: {np.min(endgame_ratios):.3f} to {np.max(endgame_ratios):.3f}")

    def get_statistics(self):
        """Get buffer statistics for monitoring"""
        if not self.buffer:
            return {}

        td_errors = [t.td_error for t in self.buffer]
        move_numbers = [t.move_number for t in self.buffer]
        game_lengths = [t.game_length for t in self.buffer]

        return {
            'size': len(self.buffer),
            'avg_td_error': np.mean(td_errors),
            'max_td_error': np.max(td_errors),
            'min_td_error': np.min(td_errors),
            'avg_move_number': np.mean(move_numbers),
            'avg_game_length': np.mean(game_lengths),
            'endgame_ratio': sum(1 for m in move_numbers if m >= 35) / len(move_numbers)
        }
#%%
def train_with_smart_buffer_sgd(online_model, replay_buffer, scaler,
                                epochs=100, batch_size=256, lr=1e-3,
                                use_prioritized=True, momentum=0.0,
                                debug_first_batch=False):
    """
    Training using SGD for transparent gradient updates.
    Perfect for debugging TD learning behavior.

    Args:
        online_model: Q-network to train
        replay_buffer: SmartReplayBuffer with pre-computed tuples
        scaler: Feature scaler
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate (fixed, no scheduling)
        use_prioritized: Whether to use prioritized sampling
        momentum: SGD momentum (0 = pure gradient descent)
        debug_first_batch: Print diagnostic info for first batch

    Returns:
        losses: List of average losses per epoch
        td_errors_history: List of average TD errors per epoch
    """
    device = next(online_model.parameters()).device

    # Pure SGD - transparent updates
    optimizer = optim.SGD(
        online_model.parameters(),
        lr=lr,
        momentum=momentum,  # 0 for pure gradient descent
        weight_decay=0      # No weight decay for clean testing
    )

    # Simple MSE loss (more transparent than SmoothL1)
    criterion = nn.MSELoss(reduction='none')

    losses = []
    td_errors_history = []

    # Store initial model state for comparison
    if debug_first_batch:
        initial_state = {k: v.clone() for k, v in online_model.state_dict().items()}

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_td = 0
        num_batches = 10

        for batch_idx in range(num_batches):
            # Sample batch
            if use_prioritized:
                beta = min(1.0, 0.4 + (epoch / epochs) * 0.6)
                batch = replay_buffer.sample_prioritized(
                    batch_size, alpha=0.6, beta=beta
                )
                if batch is None:
                    continue
                states, targets, weights, td_errors, _ = batch
                weights = weights.to(device)
            else:
                batch = replay_buffer.sample_uniform(batch_size)
                if batch is None:
                    continue
                states, targets = batch
                weights = torch.ones(batch_size).to(device)
                td_errors = torch.zeros(batch_size)

            states = states.to(device)
            targets = targets.to(device)

            # Scale features
            states_scaled = scaler.transform(states.cpu().numpy())
            states_scaled = torch.FloatTensor(states_scaled).to(device)

            # Forward pass
            predictions = online_model(states_scaled).squeeze()

            # Calculate loss
            losses_batch = criterion(predictions, targets)

            # Apply importance sampling weights if using prioritized replay
            if use_prioritized:
                loss = (losses_batch * weights).mean()
            else:
                loss = losses_batch.mean()

            # Debug first batch
            if debug_first_batch and epoch == 0 and batch_idx == 0:
                print("\n=== SGD DIAGNOSTIC: First batch ===")
                print(f"Learning rate: {lr}")
                print(f"Momentum: {momentum}")
                print(f"Weight decay: 0")

                # Before update
                with torch.no_grad():
                    pre_update_predictions = online_model(states_scaled).squeeze()

                print(f"\nFirst 5 predictions: {pre_update_predictions[:5].cpu()}")
                print(f"First 5 targets: {targets[:5].cpu()}")
                print(f"Differences: {(pre_update_predictions - targets)[:5].cpu()}")

                # Loss info
                print(f"\nRaw losses (first 5): {losses_batch[:5].cpu()}")
                if use_prioritized:
                    print(f"Importance weights (first 5): {weights[:5].cpu()}")
                print(f"Loss for backprop: {loss.item():.8f}")

                # Gradient info
                optimizer.zero_grad()
                loss.backward()

                first_param = next(online_model.parameters())
                grad_norm = first_param.grad.norm().item() if first_param.grad is not None else 0
                grad_max = first_param.grad.abs().max().item() if first_param.grad is not None else 0

                print(f"\nGradient norm: {grad_norm:.8f}")
                print(f"Gradient max: {grad_max:.8f}")

                # Update and measure change
                initial_param = first_param.clone()
                optimizer.step()
                param_change = (first_param - initial_param).abs().max().item()

                print(f"Parameter change: {param_change:.8f}")
                print(f"Expected change (lr * grad_max): {lr * grad_max:.8f}")

                # After update
                with torch.no_grad():
                    post_update_predictions = online_model(states_scaled).squeeze()
                    pred_change = (post_update_predictions - pre_update_predictions).abs().mean().item()

                print(f"Average prediction change: {pred_change:.8f}")
                print("=" * 50)
            else:
                # Normal training step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            epoch_td += td_errors.mean().item()

        # Calculate epoch statistics
        avg_loss = epoch_loss / num_batches
        avg_td = epoch_td / num_batches
        losses.append(avg_loss)
        td_errors_history.append(avg_td)

        # Progress logging
        if epoch % 10 == 0:
            stats = replay_buffer.get_statistics()
            print(f"Epoch {epoch:3d}: Loss={avg_loss:.6f}, "
                  f"AvgTD={avg_td:.4f}, BufferTD={stats['avg_td_error']:.4f}")

            if debug_first_batch and epoch == 0:
                # Check total parameter change from start
                current_state = online_model.state_dict()
                max_change = max(
                    (current_state[k] - initial_state[k]).abs().max().item()
                    for k in initial_state.keys()
                )
                print(f"  Max parameter change from start: {max_change:.8f}")

    # Final summary if debugging
    if debug_first_batch:
        print("\n=== Training Complete ===")
        print(f"Final average loss: {losses[-1]:.6f}")

        current_state = online_model.state_dict()
        for name, param in initial_state.items():
            change = (current_state[name] - param).abs().max().item()
            if change > 1e-6:
                print(f"Layer {name}: max change = {change:.8f}")

    return losses, td_errors_history
#%%

#More complex - trains with smart buffer
def train_with_smart_buffer(online_model, replay_buffer, scaler,
                           epochs=100, batch_size=256, lr=5e-4,
                           use_prioritized=True):
    """
    Simplified training using pre-computed TD errors
    """
    device = next(online_model.parameters()).device

    # Optimizer choice
    optimizer = optim.AdamW(  # AdamW often better than Adam for RL
        online_model.parameters(),
        lr=lr,
        weight_decay=1e-5,
        amsgrad=True  # More stable variant
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr/10
    )

    criterion = nn.SmoothL1Loss(reduction='none')

    losses = []
    td_errors_history = []

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_td = 0
        num_batches = 10  # Fixed batches per epoch

        for batch_idx in range(num_batches):
            if use_prioritized:
                # Anneal beta from 0.4 to 1.0
                beta = min(1.0, 0.4 + (epoch / epochs) * 0.6)
                batch = replay_buffer.sample_prioritized(
                    batch_size, alpha=0.6, beta=beta
                )
                if batch is None:
                    continue
                states, targets, weights, td_errors, _ = batch
                weights = weights.to(device)
            else:
                batch = replay_buffer.sample_uniform(batch_size)
                if batch is None:
                    continue
                states, targets = batch
                weights = torch.ones(batch_size).to(device)
                td_errors = torch.zeros(batch_size)

            states = states.to(device)
            targets = targets.to(device)

            # Scale features
            states_scaled = scaler.transform(states.cpu().numpy())
            states_scaled = torch.FloatTensor(states_scaled).to(device)

            # Forward pass
            predictions = online_model(states_scaled).squeeze()

            # Weighted loss
            losses_batch = criterion(predictions, targets)
            loss = (losses_batch * weights).mean()
            if epoch == 0 and batch_idx == 0:  # First batch only
                print("\n=== DIAGNOSTIC: First batch ===")
                # Store initial model state
                initial_param = next(online_model.parameters()).clone()

                # Check the actual loss being backpropagated
                print(f"\nLoss tensor (before mean): {losses_batch[:5].cpu()}")
                print(f"Weights for batch: {weights[:5].cpu()}")
                print(f"Weighted losses: {(losses_batch * weights)[:5].cpu()}")
                print(f"Final loss for backprop: {loss.item():.8f}")

                # Check gradients after backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online_model.parameters(), 10)
                first_layer_grad = next(online_model.parameters()).grad
                print(f"\nGradient norm: {first_layer_grad.norm().item():.8f}")
                print(f"Gradient max: {first_layer_grad.abs().max().item():.8f}")

                # After optimizer step
                optimizer.step()
                param_change = (next(online_model.parameters()) - initial_param).abs().max().item()
                print(f"Parameter change: {param_change:.8f}")
            else:
                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(online_model.parameters(), 10)

                optimizer.step()

            epoch_loss += loss.item()
            epoch_td += td_errors.mean().item()

        # Step scheduler
        scheduler.step()

        # Log progress
        avg_loss = epoch_loss / num_batches
        avg_td = epoch_td / num_batches
        losses.append(avg_loss)
        td_errors_history.append(avg_td)

        if epoch % 10 == 0:
            stats = replay_buffer.get_statistics()
            print(f"Epoch {epoch:3d}: Loss={avg_loss:.6f}, "
                  f"AvgTD={avg_td:.4f}, BufferTD={stats['avg_td_error']:.4f}, "
                  f"LR={scheduler.get_last_lr()[0]:.6f}")

    return losses, td_errors_history
#%%

#%%
# Cell that disassembles full_training_pipeline
# This particular cell takes 6000 game codes and gets ready to produce TD difference cells.
codes_file = os.path.expanduser('~/Downloads/replayMem.txt')
skip_rows = 1e6
skip_rows = int(skip_rows)
game_codes = []
with open(codes_file, 'r') as f:
    print(f"Skipping a {skip_rows} rows")
    for _ in range(skip_rows):
        f.readline()
    for counter_main, line in enumerate(f):
        if counter_main >= 6000:  # Only take first N
            break
        code = line.strip()
        game_codes.append(code)
#%%
model_path=os.path.expanduser('~/Downloads/qnet_mc_pretrained.pth')
# Load models
device_main = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(model_path, map_location=device_main)

# Initialize networks
online_model_main = QNetwork(input_dim=138).to(device_main)
online_model_main.load_state_dict(checkpoint['model_state_dict'])

target_model_main = QNetwork(input_dim=138).to(device_main)
target_model_main.load_state_dict(checkpoint['model_state_dict'])

scaler_main = checkpoint['scaler']
feature_gen = FeatureGenerator()

#%%
# Verifying to check why the training is weird:
# Generate tuples with TD errors
print("Generating training tuples with TD errors...")
training_tuples = generate_training_tuples_with_td(
    game_codes[:100],  # Just 100 games for quick test
    # game_codes, # the whole thing
    online_model_main,
    target_model_main,
    scaler_main,
    feature_gen,
    alpha=0.1,  # <-- CHANGE TO 0 for testing.
    gamma=0.99
)
#%%
replay_buffer_main = SmartReplayBuffer(capacity=100000)
for tuple_data in training_tuples:
    replay_buffer_main.push(tuple_data)
#%%
# Test with alpha=0 - model should not move - victory! We've done what we came here for.
losses_main, td_hist = train_with_smart_buffer_sgd(
    online_model_main,
    replay_buffer_main,
    scaler_main,
    epochs=100,
    lr=1e-3,
    momentum=0,  # Pure gradient descent
    debug_first_batch=False
)
#%%
def complete_td_training_loop(starting_position=1e6, total_iterations=10):
    """
    Full TD learning with refreshing buffers
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load initial models
    checkpoint = torch.load(os.path.expanduser('~/Downloads/qnet_mc_pretrained.pth'), map_location=device)
    online_model = QNetwork(input_dim=138).to(device)
    online_model.load_state_dict(checkpoint['model_state_dict'])

    target_model = QNetwork(input_dim=138).to(device)
    target_model.load_state_dict(checkpoint['model_state_dict'])

    scaler = checkpoint['scaler']
    feature_gen = FeatureGenerator()

    all_losses = []

    for iteration in range(total_iterations):
        print(f"\n{'='*50}")
        print(f"TD ITERATION {iteration+1}/{total_iterations}")
        print(f"{'='*50}")

        # 1. Load NEW games each iteration
        start_pos = int(starting_position + iteration * 10000)
        print(f"Loading games from position {start_pos}")

        game_codes = []
        codes_file = os.path.expanduser('~/Downloads/replayMem.txt')

        with open(codes_file, 'r') as f:
            for _ in range(start_pos):
                f.readline()
            for i, line in enumerate(f):
                if i >= 6000:
                    break
                game_codes.append(line.strip())

        # 2. Generate TD targets with CURRENT models
        print("Generating fresh TD targets...")
        training_tuples = generate_training_tuples_with_td(
            game_codes,
            online_model,    # Uses current online model!
            target_model,    # Uses current target model
            scaler,
            feature_gen,
            alpha=0.1,
            gamma=0.99
        )

        # 3. Create fresh buffer
        replay_buffer = SmartReplayBuffer(capacity=100000)
        for tuple_data in training_tuples:
            replay_buffer.push(tuple_data)

        stats = replay_buffer.get_statistics()
        print(f"Buffer stats: {len(replay_buffer.buffer)} samples")
        print(f"Avg TD error: {stats['avg_td_error']:.4f}")

        # 4. Train <= updated!
        losses, td_hist = train_with_smart_buffer_sgd(
            online_model,
            replay_buffer,
            scaler,
            epochs=100,
            lr=1e-3,
            momentum=0,  # Pure gradient descent
            debug_first_batch=True
        )

        all_losses.extend(losses)

        # 5. Update target network every 30 iterations
        if (iteration + 1) % 30 == 0:
            target_model.load_state_dict(online_model.state_dict())
            print("✓ Updated target network")

        # 6. Save checkpoint
        torch.save({
            'model_state_dict': online_model.state_dict(),  # The trained model
            'scaler': scaler,                               # For inference
            'iteration': iteration,                         # Track progress
        }, f'qnet_td_iter_{iteration+1}.pth')
        # 7. Optional: Quick test vs original
        if (iteration + 1) % 5 == 0:
            print("\nQuick performance check... which we are not doing")
            # Run 10 games vs original model
            # ... test code ...

    return online_model, all_losses
#%%
complete_td_training_loop()
#%%
