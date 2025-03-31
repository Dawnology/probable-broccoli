# PyTorch MCTS Chess Engine

A chess engine using Monte Carlo Tree Search (MCTS) guided by a PyTorch neural network, inspired by AlphaZero.

## Features

- Neural network-guided MCTS for strong chess play
- UCI protocol support for use with any UCI-compatible chess GUI
- Board representation using input planes for piece positions, castling rights, etc.
- Data generation from PGN files using Stockfish for high-quality training data
- Training pipeline with validation and checkpoint management

## Requirements

- Python 3.8+
- PyTorch 2.0.0+
- python-chess 1.9.0+
- NumPy 1.20.0+
- h5py 3.7.0+
- PyYAML 6.0+
- Stockfish (for data generation)

## Project Structure

```
.
├── src/                      # Core engine code
│   ├── board_utils.py        # Board representation functions
│   ├── model.py              # Neural network model
│   ├── mcts.py               # Monte Carlo Tree Search implementation
│   ├── engine.py             # Engine combining NN and MCTS
│   └── uci.py                # Universal Chess Interface handler
├── scripts/                  # Utility scripts
│   ├── generate_data.py      # Generate training data from PGN files
│   └── train.py              # Train the neural network
├── models/                   # Saved model checkpoints
├── data/                     # Training data
├── stockfish/                # Stockfish engine executable
├── pgn_training/             # PGN files for training
├── main.py                   # Main entry point
├── config.yaml               # Configuration file
└── requirements.txt          # Python dependencies
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Ensure Stockfish is installed and available at the path specified in `config.yaml`

## Usage

### Running the Chess Engine

To use the engine with any UCI-compatible chess GUI:

```
python main.py
```

Optional arguments:
- `--model MODEL_PATH`: Path to a trained model
- `--config CONFIG_PATH`: Path to a configuration file (default: `config.yaml`)
- `--name ENGINE_NAME`: Name to report to UCI
- `--author AUTHOR_NAME`: Author name to report to UCI

### Generating Training Data

To generate training data from PGN files:

```
python scripts/generate_data.py
```

Optional arguments:
- `--pgn-dir PGN_DIR`: Directory containing PGN files (default: `pgn_training`)
- `--output OUTPUT_PATH`: Output HDF5 file (default: `data/processed_training_data.h5`)
- `--stockfish STOCKFISH_PATH`: Path to Stockfish executable (default: `stockfish/stockfish.exe`)
- `--config CONFIG_PATH`: Configuration file (default: `config.yaml`)
- `--workers N`: Number of worker processes
- `--sample-rate RATE`: Fraction of positions to sample from games (0-1)
- `--max-positions N`: Maximum number of positions to process
- `--val-split RATIO`: Fraction of data to use for validation

### Training the Neural Network

To train the neural network:

```
python scripts/train.py
```

Optional arguments:
- `--data DATA_PATH`: Path to the HDF5 data file (default: `data/processed_training_data.h5`)
- `--config CONFIG_PATH`: Path to the configuration file (default: `config.yaml`)
- `--output-dir DIR`: Directory to save model checkpoints (default: `models`)
- `--epochs N`: Number of epochs to train for (default: 50)
- `--batch-size N`: Batch size for training (default: 256)
- `--lr RATE`: Initial learning rate (default: 0.001)
- `--workers N`: Number of data loading workers (default: 4)
- `--checkpoint PATH`: Path to a checkpoint to resume training from

## Configuration

Edit `config.yaml` to customize:
- Paths to Stockfish and PGN files
- Neural network parameters
- Data generation parameters
- MCTS parameters

## Training Workflow

1. Generate training data:
   ```
   python scripts/generate_data.py
   ```
2. Train the neural network:
   ```
   python scripts/train.py
   ```
3. Use the trained model in the engine:
   ```
   python main.py --model models/best_model.pt
   ```

## License

MIT License 