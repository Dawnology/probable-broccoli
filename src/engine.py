"""
Chess engine implementation that combines neural network evaluation with MCTS.
"""
import os
import time
import chess
import torch
import yaml
import random
import math

from src.model import ChessModel
from src.mcts import MCTS, Node

class Engine:
    """Chess engine combining neural network evaluation with MCTS."""
    
    def __init__(self, model_path=None, config_path=None, device=None):
        """Initialize the chess engine.
        
        Args:
            model_path: Path to saved model weights. If None, use untrained model.
            config_path: Path to configuration file. If None, use default.
            device: Device to run the model on (cpu or cuda).
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Determine device
        self.device = device
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Try to determine the correct input dimensions from model checkpoint if available
        input_planes_from_checkpoint = None
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # Try to extract input dimension from the first layer's weights
                    first_layer_weights = checkpoint['model_state_dict'].get('conv_input.weight')
                    if first_layer_weights is not None:
                        # Extract the input channels dimension (second dimension in weights tensor)
                        input_planes_from_checkpoint = first_layer_weights.shape[1]
                        print(f"Detected {input_planes_from_checkpoint} input planes from model weights")
            except Exception as e:
                print(f"Warning: Could not determine input dimensions from model: {e}")
        
        # Initialize model with configuration
        if input_planes_from_checkpoint is not None:
            # Use dimensions detected from checkpoint
            num_input_planes = input_planes_from_checkpoint
            policy_vector_size = self.config.get('policy_vector_size', 4672)
            print(f"Creating model with input planes: {num_input_planes} (from checkpoint)")
        else:
            # Use config file or defaults otherwise
            num_input_planes = self.config.get('num_input_planes', 19) * self.config.get('history_length', 8)
            policy_vector_size = self.config.get('policy_vector_size', 4672)
            print(f"Creating model with input planes: {num_input_planes} (from config file)")
        
        self.model = ChessModel(
            num_input_planes=num_input_planes,
            policy_output_size=policy_vector_size
        )
        
        # Load model weights if provided
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model.load(model_path, map_location=self.device)
        else:
            print("Using untrained model")
        
        # Initialize MCTS
        self.mcts = MCTS(
            model=self.model,
            device=self.device,
            num_simulations=self.config.get('num_simulations', 800),
            c_puct=self.config.get('c_puct', 1.0)
        )
    
    def _load_config(self, config_path):
        """Load configuration from file."""
        default_config = {
            'num_input_planes': 19,
            'policy_vector_size': 4672,
            'history_length': 8,
            'num_simulations': 800,
            'c_puct': 1.0
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return {**default_config, **config}
        else:
            return default_config
    
    def search(self, board, time_limit_ms=None, depth_limit=None, nodes_limit=None, temperature=0.0):
        """
        Search for the best move in the current position.
        
        Args:
            board: The chess.Board to search from.
            time_limit_ms: Maximum search time in milliseconds.
            depth_limit: Maximum search depth (not used in MCTS).
            nodes_limit: Maximum number of nodes to search (overrides default simulations if set).
            temperature: Temperature for move selection (0.0 means deterministic best move).
            
        Returns:
            chess.Move: The best move found.
        """
        try:
            # Make a deep copy of the board to ensure we're not affecting the original
            search_board = board.copy()
            
            # Print board state info for debugging
            print(f"info string Searching position: {search_board.fen()}")
            print(f"info string Legal moves: {len(list(search_board.legal_moves))}")
            
            # Initialize search parameters
            num_simulations = self.mcts.num_simulations
            if nodes_limit:
                num_simulations = nodes_limit
                self.mcts.num_simulations = nodes_limit
            
            # Time-limited search
            if time_limit_ms:
                start_time = time.time()
                end_time = start_time + (time_limit_ms / 1000.0)
                
                # Create root node once
                root = Node(search_board)
                
                # Run simulations until time is up
                simulation_count = 0
                last_info_time = start_time
                
                while time.time() < end_time and simulation_count < num_simulations:
                    self.mcts.simulate(root)
                    simulation_count += 1
                    
                    # Periodically report search progress (every 250ms)
                    current_time = time.time()
                    elapsed_ms = int((current_time - start_time) * 1000)
                    
                    if simulation_count % 10 == 0 and current_time - last_info_time > 0.25:
                        last_info_time = current_time
                        
                        # Get current best move and score
                        if root.children:
                            # Find best child based on visit counts
                            best_move, best_child = max(root.children.items(), key=lambda x: x[1].visit_count)
                            
                            # Calculate score in centipawns (from perspective of side to move)
                            # Map the value from [-1, 1] to centipawns with a reasonable scaling factor
                            cp_score = int(best_child.q_value * 100)
                            
                            # Get principal variation (best moves for both sides)
                            pv_moves = self._extract_pv(root, max_length=5)
                            pv_str = " ".join([move.uci() for move in pv_moves])
                            
                            # Calculate effective depth based on simulations
                            effective_depth = max(1, int(math.log2(simulation_count + 1)))
                            
                            # Output in the Stockfish-like format
                            nps = int(simulation_count / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0
                            print(f"info depth {effective_depth} score cp {cp_score} nodes {simulation_count} " + 
                                  f"time {elapsed_ms} nps {nps} pv {pv_str}")
                            
                            # Also output WDL probabilities periodically
                            if simulation_count % 20 == 0:
                                # Convert Q-value to win probability
                                win_prob = (best_child.q_value + 1) / 2  # Scale from [-1,1] to [0,1]
                                draw_prob = 0.15  # Approximate draw probability
                                loss_prob = 1 - win_prob - draw_prob
                                
                                # Clamp to valid ranges
                                win_prob = max(0, min(1, win_prob))
                                loss_prob = max(0, min(1, loss_prob))
                                draw_prob = max(0, min(1, 1 - win_prob - loss_prob))
                                
                                # Format as per UCI WDL convention (1000-based)
                                wdl_win = int(win_prob * 1000)
                                wdl_draw = int(draw_prob * 1000)
                                wdl_loss = int(loss_prob * 1000)
                                
                                print(f"info wdl {wdl_win} {wdl_draw} {wdl_loss}")
                
                # Send final search information
                elapsed_ms = int((time.time() - start_time) * 1000)
                print(f"info string Completed {simulation_count} simulations in {elapsed_ms:.1f}ms")
                
                if root.children:
                    best_move, best_child = max(root.children.items(), key=lambda x: x[1].visit_count)
                    cp_score = int(best_child.q_value * 100)
                    pv_moves = self._extract_pv(root, max_length=5)
                    pv_str = " ".join([move.uci() for move in pv_moves])
                    nps = int(simulation_count / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0
                    
                    # Send final detailed info
                    hashfull = min(100, int((len(root.children) / 1000.0) * 100))  # Simulate hash fullness
                    print(f"info depth {simulation_count//5} score cp {cp_score} nodes {simulation_count} " + 
                          f"time {elapsed_ms} nps {nps} hashfull {hashfull} pv {pv_str}")
                          
                    # Add WDL stats (win/draw/loss) estimation
                    if best_child.visit_count > 0:
                        win_prob = (best_child.q_value + 1) / 2  # Convert [-1,1] to [0,1]
                        draw_prob = 0.2  # Fixed estimate for draw probability
                        loss_prob = 1 - win_prob - draw_prob
                        
                        # Clamp probabilities to valid ranges
                        win_prob = max(0, min(1, win_prob))
                        loss_prob = max(0, min(1, loss_prob))
                        draw_prob = max(0, min(1, 1 - win_prob - loss_prob))
                        
                        # Scale to 1000-based WDL format
                        wdl_win = int(win_prob * 1000)
                        wdl_draw = int(draw_prob * 1000)
                        wdl_loss = int(loss_prob * 1000)
                        
                        print(f"info wdl {wdl_win} {wdl_draw} {wdl_loss}")
                
                return self.mcts.select_move(root, temperature)
            else:
                # Fixed number of simulations
                start_time = time.time()
                
                # Create root node
                root = Node(search_board)
                
                # For fixed simulations, report progress every 10%
                report_interval = max(1, num_simulations // 10)
                
                for i in range(num_simulations):
                    self.mcts.simulate(root)
                    
                    # Report progress periodically
                    if (i + 1) % report_interval == 0 or i == num_simulations - 1:
                        elapsed_ms = int((time.time() - start_time) * 1000)
                        
                        if root.children:
                            best_move, best_child = max(root.children.items(), key=lambda x: x[1].visit_count)
                            cp_score = int(best_child.q_value * 100)
                            pv_moves = self._extract_pv(root, max_length=5)
                            pv_str = " ".join([move.uci() for move in pv_moves])
                            nps = int(i / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0
                            
                            # Calculate effective depth based on simulations
                            effective_depth = int(math.log2(i + 1)) + 1
                            
                            print(f"info depth {effective_depth} score cp {cp_score} nodes {i+1} " + 
                                  f"time {elapsed_ms} nps {nps} pv {pv_str}")
                
                # Final report
                elapsed_ms = int((time.time() - start_time) * 1000)
                print(f"info string Completed {num_simulations} simulations in {elapsed_ms:.1f}ms")
                
                # Send detailed final search information
                if root.children:
                    best_move, best_child = max(root.children.items(), key=lambda x: x[1].visit_count)
                    
                    # Format score
                    cp_score = int(best_child.q_value * 100)
                    side_to_move = "BLACK" if search_board.turn == chess.BLACK else "WHITE"
                    
                    # Get principal variation for final report
                    pv_moves = self._extract_pv(root, max_length=6)
                    pv_str = " ".join([move.uci() for move in pv_moves])
                    
                    # Calculate effective depth
                    effective_depth = max(1, int(math.log2(num_simulations + 1)))
                    
                    # Calculate nps
                    nps = int(num_simulations / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0
                    
                    # Send final detailed info
                    print(f"info depth {effective_depth} score cp {cp_score} nodes {num_simulations} " + 
                          f"time {elapsed_ms} nps {nps} pv {pv_str}")
                    
                    # Add WDL stats
                    win_prob = (best_child.q_value + 1) / 2  # Convert [-1,1] to [0,1]
                    draw_prob = 0.15  # Fixed estimate for draw probability
                    loss_prob = 1 - win_prob - draw_prob
                    
                    # Clamp probabilities
                    win_prob = max(0, min(1, win_prob))
                    loss_prob = max(0, min(1, loss_prob))
                    draw_prob = max(0, min(1, 1 - win_prob - loss_prob))
                    
                    # Scale to 1000-based format
                    wdl_win = int(win_prob * 1000)
                    wdl_draw = int(draw_prob * 1000)
                    wdl_loss = int(loss_prob * 1000)
                    
                    print(f"info wdl {wdl_win} {wdl_draw} {wdl_loss}")
                    print(f"info string PovScore(Cp({cp_score}), {side_to_move})")
                
                # Return the best move
                return self.mcts.run(search_board, temperature)
        except Exception as e:
            print(f"info string Error in search: {e}")
            import traceback
            print(f"info string Traceback: {traceback.format_exc()}")
            
            # Return a valid move as fallback
            legal_moves = list(board.legal_moves)
            if legal_moves:
                selected_move = random.choice(legal_moves)
                print(f"info string Fallback: randomly selected {selected_move.uci()}")
                return selected_move
            return None
            
    def _extract_pv(self, root, max_length=5):
        """
        Extract the principal variation (sequence of best moves) from the search tree.
        
        Args:
            root: The root Node of the search tree.
            max_length: Maximum number of moves to include in the PV.
            
        Returns:
            list: A list of chess.Move objects representing the principal variation.
        """
        pv = []
        current = root
        
        # Follow the most visited path in the tree
        for _ in range(max_length):
            if not current.children:
                break
                
            # Find the child with the highest visit count
            best_move, best_child = max(current.children.items(), key=lambda x: x[1].visit_count)
            
            # Add the move to the PV
            pv.append(best_move)
            
            # Continue to the next node
            current = best_child
            
            # Stop if we've reached a terminal position
            if current.is_terminal:
                break
        
        return pv 