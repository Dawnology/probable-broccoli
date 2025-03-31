"""
Monte Carlo Tree Search (MCTS) implementation for chess.
"""
import math
import random
import numpy as np
import chess
import torch

from src.board_utils import board_to_input_planes, move_to_policy_index, policy_index_to_move, input_planes_to_tensor

class Node:
    """Node in the MCTS tree."""
    
    def __init__(self, board, parent=None, move=None, prior=0.0):
        """Initialize a new node.
        
        Args:
            board: A chess.Board object representing the current position.
            parent: The parent Node.
            move: The chess.Move that led to this node.
            prior: Prior probability assigned to this node by the policy network.
        """
        self.board = board
        self.parent = parent
        self.move = move  # Move that led to this position
        self.children = {}  # Map from moves to child nodes
        
        # MCTS statistics
        self.visit_count = 0
        self.value_sum = 0.0  # Sum of values from all simulations
        self.prior = prior
        
        # Lazily computed properties
        self._is_terminal = None
        self._terminal_value = None
        self._untried_moves = None
    
    @property
    def is_terminal(self):
        """Check if this node represents a terminal game state."""
        if self._is_terminal is None:
            self._is_terminal = self.board.is_game_over(claim_draw=True)
        return self._is_terminal
    
    @property
    def terminal_value(self):
        """Get the value of the terminal state (-1, 0, 1)."""
        if not self.is_terminal:
            return None
            
        if self._terminal_value is None:
            result = self.board.result(claim_draw=True)
            if result == '1-0':
                # White wins (1 if white to move, -1 if black to move)
                self._terminal_value = 1.0
            elif result == '0-1':
                # Black wins (-1 if white to move, 1 if black to move)
                self._terminal_value = -1.0
            else:
                # Draw
                self._terminal_value = 0.0
                
            # Adjust based on whose perspective we're calculating from
            # If it's black's turn, negate the value
            if not self.board.turn:
                self._terminal_value = -self._terminal_value
                
        return self._terminal_value
    
    @property
    def untried_moves(self):
        """Get the list of moves that have not been tried yet from this node."""
        if self._untried_moves is None:
            self._untried_moves = list(move for move in self.board.legal_moves
                                     if move not in self.children)
        return self._untried_moves
    
    @property
    def q_value(self):
        """Get the mean action value of this node (Q = value_sum / visit_count)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
        
    @property
    def ucb_score(self, c_puct=1.0):
        """Calculate the UCB (Upper Confidence Bound) score for this node."""
        if self.visit_count == 0:
            return float('inf')
            
        # AlphaZero's PUCT (Predictor + UCT) formula
        # Q + c_puct * P * sqrt(sum(N)) / (1 + N)
        parent_visit_count_total = self.parent.visit_count
        exploration_term = c_puct * self.prior * math.sqrt(parent_visit_count_total) / (1 + self.visit_count)
        return self.q_value + exploration_term
    
    def select_child(self, c_puct=1.0):
        """Select the child with the highest UCB score."""
        if not self.children:
            return None
            
        # Find the move with the highest UCB score
        best_move = max(self.children.items(),
                      key=lambda item: item[1].ucb_score)
        return best_move[0], best_move[1]
    
    def expand(self, policy_probs):
        """
        Expand the node by creating all possible child nodes.
        
        Args:
            policy_probs: A dict mapping moves to their prior probabilities.
        """
        # Create a child node for each legal move
        for move in self.board.legal_moves:
            if move not in self.children:
                # Apply the move to get the new position
                new_board = self.board.copy()
                new_board.push(move)
                
                # Get the prior probability for this move
                prior = policy_probs.get(move, 0.0)
                
                # Create a new child node
                self.children[move] = Node(new_board, parent=self, move=move, prior=prior)
                
                # Remove the move from untried moves if we're tracking them
                if self._untried_moves is not None and move in self._untried_moves:
                    self._untried_moves.remove(move)
    
    def update(self, value):
        """
        Update the node statistics with the simulation result.
        
        Args:
            value: The value to back up from simulation (-1 to 1).
        """
        self.visit_count += 1
        self.value_sum += value
        
    def __str__(self):
        """String representation of the node."""
        return f"Node(move={self.move}, visits={self.visit_count}, value={self.q_value:.3f}, prior={self.prior:.3f})"


class MCTS:
    """Monte Carlo Tree Search using a neural network for policy and value estimation."""
    
    def __init__(self, model, device=None, num_simulations=800, c_puct=1.0, dirichlet_alpha=0.3, dirichlet_weight=0.25):
        """Initialize the MCTS search.
        
        Args:
            model: The neural network model to use for policy and value prediction.
            device: The PyTorch device to run the model on.
            num_simulations: Number of simulations to run per search.
            c_puct: Exploration constant in the PUCT formula.
            dirichlet_alpha: Alpha parameter for Dirichlet noise distribution.
            dirichlet_weight: Weight of the Dirichlet noise.
        """
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode
        
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
    
    def run(self, board, temperature=1.0):
        """
        Run the MCTS algorithm on the current board position and return the best move.
        
        Args:
            board: A chess.Board object.
            temperature: Temperature parameter for move selection.
            
        Returns:
            chess.Move: The selected best move.
        """
        # Create root node
        root = Node(board)
        
        # Run simulations
        for _ in range(self.num_simulations):
            self.simulate(root)
            
        # Select the move based on the visit counts and temperature
        return self.select_move(root, temperature)
    
    def simulate(self, node):
        """
        Run a single simulation from the given node.
        
        Args:
            node: The root Node to start the simulation from.
        """
        # Phase 1: Selection - traverse the tree to a leaf node
        path = []
        current = node
        
        while current.children and not current.is_terminal:
            # Select the child with the highest UCB score
            move, current = current.select_child(self.c_puct)
            path.append(current)
            
        # Phase 2: Expansion and Evaluation
        value = 0.0
        
        if current.is_terminal:
            # Game is over in this node
            value = current.terminal_value
        else:
            # Expand the node using the policy network
            policy_probs, value = self.evaluate(current.board)
            
            # Add Dirichlet noise to the root node during search
            if current == node:
                self.add_dirichlet_noise(policy_probs)
                
            # Create children for all legal moves
            current.expand(policy_probs)
            
        # Phase 3: Backup - update statistics up the tree
        # Note: value is from the perspective of the player who just moved,
        # but we update from the perspective of the current player at each node
        for n in reversed(path):
            # Negate the value because it's from opponent's perspective
            value = -value
            n.update(value)
    
    def evaluate(self, board):
        """
        Evaluate a board position using the neural network.
        
        Args:
            board: A chess.Board object.
            
        Returns:
            tuple: (policy_probs, value) where:
                - policy_probs is a dict mapping moves to their probabilities
                - value is a float in range [-1, 1]
        """
        try:
            # Convert board to input planes - get the right number of input planes
            input_planes = board_to_input_planes(board)
            
            # Get the expected number of input planes from the model's first layer
            expected_planes = self.model.conv_input.in_channels
            
            # Handle dimension mismatch
            if input_planes.shape[0] != expected_planes:
                # If we have fewer planes than expected, pad with zeros
                if input_planes.shape[0] < expected_planes:
                    padding_needed = expected_planes - input_planes.shape[0]
                    padded_planes = np.zeros((expected_planes, 8, 8), dtype=np.float32)
                    padded_planes[:input_planes.shape[0]] = input_planes
                    input_planes = padded_planes
                # If we have more planes than expected, truncate
                else:
                    input_planes = input_planes[:expected_planes]
            
            x = torch.from_numpy(input_planes).float().unsqueeze(0)  # Add batch dimension
            x = x.to(self.device)
            
            # Evaluate the position using the neural network
            with torch.no_grad():
                policy_logits, value = self.model(x)
                policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
                value = value.item()
            
            # Convert policy vector to a dictionary of {move: probability}
            moves_probs = {}
            for move in board.legal_moves:
                move_idx = move_to_policy_index(move)
                if 0 <= move_idx < len(policy_probs):
                    moves_probs[move] = policy_probs[move_idx]
                else:
                    # Handle the case where move_idx is out of bounds
                    moves_probs[move] = 0.0
                    
            # Normalize the probabilities to sum to 1 over legal moves
            total_prob = sum(moves_probs.values())
            if total_prob > 0:
                for move in moves_probs:
                    moves_probs[move] /= total_prob
            
            return moves_probs, value
            
        except Exception as e:
            print(f"info string Evaluation error: {e}")
            
            # Return uniform probabilities as fallback
            moves_probs = {move: 1.0/len(list(board.legal_moves)) for move in board.legal_moves}
            return moves_probs, 0.0
    
    def add_dirichlet_noise(self, policy_probs):
        """
        Add Dirichlet noise to the policy probabilities at the root node.
        
        Args:
            policy_probs: Dict mapping moves to their probabilities.
        """
        # Generate Dirichlet noise
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(policy_probs))
        
        # Add noise to the probabilities
        moves = list(policy_probs.keys())
        for i, move in enumerate(moves):
            policy_probs[move] = (1 - self.dirichlet_weight) * policy_probs[move] + self.dirichlet_weight * noise[i]
    
    def select_move(self, root, temperature=1.0):
        """
        Select a move based on the visit counts of the root's children.
        
        Args:
            root: The root Node of the search tree.
            temperature: Controls the exploration/exploitation trade-off.
                - temperature -> 0: Always choose the best move (exploitation)
                - temperature -> inf: Choose moves randomly (exploration)
                
        Returns:
            chess.Move: The selected move.
        """
        # If there are no children (no legal moves), return None
        if not root.children:
            return None
            
        # Get visit counts of all children
        visits = {move: child.visit_count for move, child in root.children.items()}
        
        if temperature == 0:
            # Deterministic selection: choose the move with highest visit count
            return max(visits.items(), key=lambda x: x[1])[0]
        else:
            # Stochastic selection based on visit count distribution
            visits_temp = {move: count ** (1 / temperature) for move, count in visits.items()}
            total = sum(visits_temp.values())
            
            # Normalize to get probabilities
            probs = {move: count / total for move, count in visits_temp.items()}
            
            # Choose a move based on the probability distribution
            moves = list(probs.keys())
            probabilities = list(probs.values())
            
            return random.choices(moves, weights=probabilities, k=1)[0] 