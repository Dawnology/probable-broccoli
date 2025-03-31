#!/usr/bin/env python3
"""
Data generation script for chess AI training.

This script:
1. Reads chess games from PGN files
2. Extracts positions from the games
3. Evaluates positions with Stockfish to create labeled training data
4. Stores the processed data in HDF5 format for training
"""
import os
import sys
import time
import argparse
import glob
import random
import multiprocessing as mp
import chess
import chess.pgn
import chess.engine
import h5py
import numpy as np
import yaml
import math
from tqdm import tqdm

# Add the parent directory to the Python path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.board_utils import board_to_input_planes, move_to_policy_index

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate training data from PGN files using Stockfish")
    parser.add_argument("--pgn-dir", type=str, default="pgn_training",
                        help="Directory containing PGN files")
    parser.add_argument("--output", type=str, default="data/processed_training_data.h5",
                        help="Output HDF5 file")
    parser.add_argument("--stockfish", type=str, default="stockfish/stockfish.exe",
                        help="Path to Stockfish executable")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Configuration file")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of worker processes")
    parser.add_argument("--sample-rate", type=float, default=0.05,
                        help="Fraction of positions to sample from games (0-1)")
    parser.add_argument("--max-positions", type=int, default=1000000,
                        help="Maximum number of positions to process")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of data to use for validation")
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        print(f"Warning: Configuration file {config_path} not found. Using defaults.")
        return {}
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_game_positions(pgn_path, sample_rate=0.05, max_positions=None):
    """
    Extract a sample of positions from PGN games.
    
    Args:
        pgn_path: Path to PGN file
        sample_rate: Fraction of positions to extract from each game (0-1)
        max_positions: Maximum number of positions to extract in total
        
    Returns:
        list: A list of (board, result) tuples
    """
    positions = []
    num_games = 0
    
    # Open PGN file
    with open(pgn_path, 'r') as f:
        while True:
            # Read the next game
            game = chess.pgn.read_game(f)
            if game is None:
                break
                
            num_games += 1
            
            # Parse game result
            result_str = game.headers.get("Result", "*")
            if result_str == "1-0":
                result = 1.0  # White wins
            elif result_str == "0-1":
                result = -1.0  # Black wins
            else:
                result = 0.0  # Draw or unknown
            
            # Extract positions from the game
            board = game.board()
            move_count = 0
            
            for move in game.mainline_moves():
                # Skip early moves (usually opening theory)
                if move_count < 5:
                    board.push(move)
                    move_count += 1
                    continue
                
                # Randomly sample positions based on sample_rate
                if random.random() < sample_rate:
                    positions.append((board.copy(), result))
                
                # Apply the move
                board.push(move)
                move_count += 1
                
                # Check if we've reached the maximum number of positions
                if max_positions and len(positions) >= max_positions:
                    return positions
    
    print(f"Extracted {len(positions)} positions from {num_games} games in {pgn_path}")
    return positions

def evaluate_position(board, engine, limit):
    """
    Evaluate a position using Stockfish.
    
    Args:
        board: A chess.Board object
        engine: A chess.engine.SimpleEngine instance
        limit: Search limit (e.g., {"depth": 10})
        
    Returns:
        tuple: (value, top_moves) where:
            - value is a float in range [-1, 1]
            - top_moves is a list of (move, probability) pairs
    """
    # Skip evaluation for terminal positions
    if board.is_game_over(claim_draw=True):
        # Determine the game result
        result = board.result(claim_draw=True)
        if result == "1-0":
            value = 1.0
        elif result == "0-1":
            value = -1.0
        else:
            value = 0.0
        
        # Adjust value based on whose turn it is
        if not board.turn:  # Black to move
            value = -value
            
        # For terminal positions, there are no legal moves
        return value, []
    
    # Run the analysis with Stockfish
    multipv = limit.pop("multipv", 5)
    result = engine.analyse(board, chess.engine.Limit(**limit), multipv=multipv)
    limit["multipv"] = multipv  # Restore multipv
    
    # Extract the score for the main line
    score = result[0]["score"].relative.score(mate_score=10000)
    
    # Convert score to a value in range [-1, 1]
    # Using tanh to scale centipawn values to [-1, 1]
    if score is not None:
        value = math.tanh(score / 600.0)  # Scaling factor 600 centipawns
    else:
        # Handle case where score is None (shouldn't happen)
        value = 0.0
    
    # Extract the top moves and assign probabilities
    # For simplicity, use softmax on the scores
    top_moves = []
    scores = []
    
    for info in result:
        # Skip if no moves in PV
        if not info.get("pv"):
            continue
            
        # Get the move and score
        move = info["pv"][0]
        s = info["score"].relative.score(mate_score=10000)
        
        if s is not None:
            top_moves.append(move)
            scores.append(s)
    
    # Convert scores to probabilities using softmax
    probs = softmax(scores)
    
    # Combine moves and probabilities
    top_moves_with_probs = [(move, prob) for move, prob in zip(top_moves, probs)]
    
    return value, top_moves_with_probs

def softmax(x):
    """Compute softmax values for the scores."""
    # Subtract max for numerical stability
    if not x:
        return []
    e_x = np.exp(np.array(x) - np.max(x))
    return e_x / e_x.sum()

def worker_init(stockfish_path, limit, dtype=np.float32):
    """Initialize a worker process with its own Stockfish instance."""
    if not os.path.exists(stockfish_path):
        raise FileNotFoundError(f"Stockfish executable not found at: {stockfish_path}")
        
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except Exception as e:
        raise Exception(f"Failed to start Stockfish engine: {str(e)}")
    
    # Store the engine and search parameters in global variables for this process
    global WORKER_ENGINE, WORKER_LIMIT, WORKER_DTYPE
    WORKER_ENGINE = engine
    WORKER_LIMIT = limit
    WORKER_DTYPE = dtype
    
    return engine

def worker_cleanup():
    """Cleanup resources used by the worker."""
    global WORKER_ENGINE
    if WORKER_ENGINE:
        WORKER_ENGINE.quit()

def process_position(args):
    """
    Process a single position (to be called by worker processes).
    
    Args:
        args: Tuple of (board, result, idx)
        
    Returns:
        tuple: (idx, input_planes, policy_target, value_target)
    """
    board, result, idx = args
    
    # Access the global engine instance and search parameters
    global WORKER_ENGINE, WORKER_LIMIT, WORKER_DTYPE
    
    try:
        # Evaluate the position with Stockfish
        value, top_moves_with_probs = evaluate_position(board, WORKER_ENGINE, dict(WORKER_LIMIT))
        
        # Create input planes from the board
        input_planes = board_to_input_planes(board)
        
        # Create policy target (vector of move probabilities)
        policy_target = np.zeros(4672, dtype=WORKER_DTYPE)  # Standard UCI move representation size
        
        for move, prob in top_moves_with_probs:
            policy_idx = move_to_policy_index(move)
            if 0 <= policy_idx < len(policy_target):
                policy_target[policy_idx] = prob
        
        # If no legal moves found, distribute probability equally among legal moves
        if not top_moves_with_probs and not board.is_game_over():
            legal_moves = list(board.legal_moves)
            for move in legal_moves:
                policy_idx = move_to_policy_index(move)
                if 0 <= policy_idx < len(policy_target):
                    policy_target[policy_idx] = 1.0 / len(legal_moves)
        
        # Return values
        return idx, input_planes, policy_target, value
        
    except Exception as e:
        print(f"Error processing position {idx}: {str(e)}")
        # Return empty data for failed positions
        return idx, None, None, None

def create_dataset(output_path, positions, stockfish_path, config, workers=4, val_split=0.1):
    """
    Process positions with Stockfish and create HDF5 dataset.
    
    Args:
        output_path: Path to output HDF5 file
        positions: List of (board, result) tuples
        stockfish_path: Path to Stockfish executable
        config: Configuration dictionary
        workers: Number of worker processes
        val_split: Fraction of data to use for validation
    
    Returns:
        int: Number of positions successfully processed
    """
    # Calculate total positions to process
    total_positions = len(positions)
    print(f"Processing {total_positions} positions with {workers} workers")
    
    # Prepare search parameters for Stockfish
    limit = {
        "depth": config.get("stockfish_limit", {}).get("depth", 10),
        "multipv": config.get("stockfish_multipv", 5)
    }
    
    # Initialize output arrays
    num_input_planes = config.get("num_input_planes", 19)
    policy_vector_size = config.get("policy_vector_size", 4672)
    
    # Start worker processes
    with mp.Pool(workers, initializer=worker_init, initargs=(stockfish_path, limit)) as pool:
        try:
            # Process positions with tqdm progress bar
            results = []
            
            # Add indices to positions for tracking
            indexed_positions = [(board, result, i) for i, (board, result) in enumerate(positions)]
            
            # Process in chunks for better progress tracking
            chunk_size = 100  # Process 100 positions per chunk
            for i in tqdm(range(0, len(indexed_positions), chunk_size)):
                chunk = indexed_positions[i:i+chunk_size]
                chunk_results = list(pool.map(process_position, chunk))
                results.extend(chunk_results)
                
            # Filter out failed positions
            results = [r for r in results if r[1] is not None]
            
            # Sort results by index
            results.sort(key=lambda x: x[0])
            
            # Extract data arrays
            input_planes_list = [r[1] for r in results]
            policy_targets = [r[2] for r in results]
            value_targets = [r[3] for r in results]
            
            # Check if we have any valid results
            if not results:
                print("No valid positions were processed. Exiting.")
                return 0
            
            # Determine splits for training and validation sets
            n_val = int(len(results) * val_split)
            n_train = len(results) - n_val
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write data to HDF5 file
            with h5py.File(output_path, 'w') as f:
                # Create datasets
                f.create_dataset('train_x', data=np.array(input_planes_list[:n_train]), compression='gzip')
                f.create_dataset('train_policy', data=np.array(policy_targets[:n_train]), compression='gzip')
                f.create_dataset('train_value', data=np.array(value_targets[:n_train]), compression='gzip')
                
                if n_val > 0:
                    f.create_dataset('val_x', data=np.array(input_planes_list[n_train:]), compression='gzip')
                    f.create_dataset('val_policy', data=np.array(policy_targets[n_train:]), compression='gzip')
                    f.create_dataset('val_value', data=np.array(value_targets[n_train:]), compression='gzip')
                
                # Store metadata
                f.attrs['num_positions'] = len(results)
                f.attrs['num_train'] = n_train
                f.attrs['num_val'] = n_val
                f.attrs['num_input_planes'] = num_input_planes
                f.attrs['policy_vector_size'] = policy_vector_size
                f.attrs['stockfish_depth'] = limit['depth']
                f.attrs['stockfish_multipv'] = limit['multipv']
                
            print(f"Successfully created dataset with {n_train} training and {n_val} validation positions")
            return len(results)
            
        except KeyboardInterrupt:
            print("Interrupted by user. Stopping workers...")
            pool.terminate()
            pool.join()
            return 0
        finally:
            # Ensure all workers cleanup properly
            for _ in range(workers):
                pool.apply_async(worker_cleanup)

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Verify Stockfish path
    stockfish_path = args.stockfish
    if not os.path.exists(stockfish_path):
        print(f"Error: Stockfish executable not found at {stockfish_path}")
        print("Please specify the correct path using --stockfish argument")
        return 1
        
    # Find all PGN files in the directory
    pgn_files = []
    if os.path.isdir(args.pgn_dir):
        pgn_files = glob.glob(os.path.join(args.pgn_dir, "*.pgn"))
    elif os.path.isfile(args.pgn_dir) and args.pgn_dir.endswith('.pgn'):
        pgn_files = [args.pgn_dir]
    
    if not pgn_files:
        print(f"No PGN files found in {args.pgn_dir}")
        return 1
    
    print(f"Found {len(pgn_files)} PGN files")
    print(f"Parsing positions from PGN files...")
    
    # Shuffle the PGN files to mix games from different sources
    random.shuffle(pgn_files)
    
    # Extract positions from PGN files
    positions = []
    max_positions_per_file = args.max_positions // len(pgn_files) if args.max_positions else None
    
    for pgn_file in pgn_files:
        # Extract positions from this file
        file_positions = extract_game_positions(
            pgn_file,
            sample_rate=args.sample_rate,
            max_positions=max_positions_per_file
        )
        positions.extend(file_positions)
        
        # Check if we've reached the maximum number of positions
        if args.max_positions and len(positions) >= args.max_positions:
            positions = positions[:args.max_positions]
            break
    
    print(f"Extracted {len(positions)} total positions")
    
    # If no positions were extracted, exit
    if not positions:
        print("No positions to process. Exiting.")
        return 1
    
    # Process positions and create dataset
    processed_count = create_dataset(
        args.output,
        positions,
        args.stockfish,
        config,
        workers=args.workers,
        val_split=args.val_split
    )
    
    if processed_count > 0:
        print(f"Successfully processed {processed_count} positions")
        return 0
    else:
        print("Data generation failed")
        return 1

if __name__ == "__main__":
    exit(main()) 