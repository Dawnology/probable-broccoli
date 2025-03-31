#!/usr/bin/env python3
"""
Main entry point for chess engine - handles command line arguments and starts UCI loop.
"""
import os
import argparse

from src.engine import Engine
from src.uci import UCIHandler

def main():
    """Parse arguments and start UCI communication."""
    parser = argparse.ArgumentParser(description="PyTorch MCTS Chess Engine")
    parser.add_argument("--model", type=str, default=None, 
                        help="Path to the model weights file")
    parser.add_argument("--config", type=str, default="config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--name", type=str, default="PyTorch MCTS Chess",
                        help="Engine name to report to UCI")
    parser.add_argument("--author", type=str, default="AI Developer",
                        help="Engine author to report to UCI")
    args = parser.parse_args()
    
    # Resolve relative paths
    if args.config:
        args.config = os.path.abspath(args.config)
    if args.model:
        args.model = os.path.abspath(args.model)
    
    # Initialize engine with model if provided
    print(f"Initializing engine with config: {args.config}")
    if args.model:
        print(f"Loading model from: {args.model}")
    
    try:
        engine = Engine(
            model_path=args.model,
            config_path=args.config
        )
        
        # Initialize and start UCI handler
        uci_handler = UCIHandler(
            engine=engine,
            name=args.name,
            author=args.author
        )
        
        print(f"Starting UCI loop for {args.name}")
        uci_handler.uci_loop()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main()) 