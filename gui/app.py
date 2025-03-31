#
# Web based GUI for BBC chess engine
#

# packages
from flask import Flask
from flask import render_template
from flask import request
import chess
import chess.engine
import chess.pgn
import io
import random
from flask import jsonify
from flask import Response
from flask_pymongo import PyMongo
from datetime import datetime
import json
import subprocess
import threading
import time
import os
import sys

# Add parent directory to path so we can import our engine modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# create web app instance
app = Flask(__name__)

# Path to mcts engine
MCTS_ENGINE_PATH = os.path.join(os.path.dirname(__file__), '..', 'main.py')

# Engine choice
ENGINE_CHOICES = {
    'bbc': './gui/engine/bbc_1.4_sf_nnue_final_64bit_windows.exe',
    'mcts': MCTS_ENGINE_PATH
}

# probe book move
def probe_book(pgn):
    # open book file
    with open('./gui/engine/book.txt') as f:
        # read book games
        book = f.read()

        # init board        
        board = chess.Board()
        
        # define response moves
        response_moves = []

        # loop over book lines
        for line in book.split('\n')[0:-1]:
            # define variation
            variation = []
            
            # loop over line moves
            for move in line.split():
                variation.append(chess.Move.from_uci(move))
            
            # parse variation to SAN
            san = board.variation_san(variation)
            
            # match book line line
            if pgn in san:
                try:
                    # black move
                    if san.split(pgn)[-1].split()[0][0].isdigit():
                        response_moves.append(san.split(pgn)[-1].split()[1])
                    
                    # white move
                    else:
                        response_moves.append(san.split(pgn)[-1].split()[0])
                
                except:
                    pass
            
            # engine makes first move
            if pgn == '':
                response_moves.append(san.split('1. ')[-1].split()[0])

        # return random response move
        if len(response_moves):
            print('BOOK MOVE:', random.choice(response_moves))
            return random.choice(response_moves)
        
        else:
            return 0

# Custom PyTorch MCTS UCI adapter
class PyTorchMCTSEngine:
    """A UCI wrapper for our PyTorch MCTS engine"""
    
    def __init__(self):
        self.process = None
        self.start_engine()
        
    def start_engine(self):
        """Start the engine process"""
        self.process = subprocess.Popen(
            ['python', MCTS_ENGINE_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1  # Line buffered
        )
        
        # Initialize UCI
        self._send_command('uci')
        self._wait_for('uciok')
        
        # Make sure engine is ready
        self._send_command('isready')
        self._wait_for('readyok')
    
    def _send_command(self, command):
        """Send a command to the engine"""
        if self.process and self.process.poll() is None:
            print(f"Sending to engine: {command}")
            self.process.stdin.write(f"{command}\n")
            self.process.stdin.flush()
    
    def _wait_for(self, text, timeout=10):
        """Wait for a specific text in the engine output"""
        start_time = time.time()
        output = []
        
        while time.time() - start_time < timeout:
            line = self.process.stdout.readline().strip()
            if line:
                print(f"Engine: {line}")
                output.append(line)
                if text in line:
                    return output
        
        raise TimeoutError(f"Timeout waiting for '{text}' from engine. Got: {output}")
    
    def _read_until_bestmove(self, timeout=60):
        """Read engine output until a bestmove is found"""
        start_time = time.time()
        output = []
        info = {
            'depth': 0,
            'seldepth': 0,
            'score': 0,
            'nodes': 0,
            'time': 0,
            'nps': 0,
            'hashfull': 0,
            'pv': [],
            'wdl': None
        }
        
        while time.time() - start_time < timeout:
            line = self.process.stdout.readline().strip()
            if not line:
                time.sleep(0.001)  # Small delay to prevent CPU hogging
                continue
                
            print(f"Engine: {line}")
            output.append(line)
            
            # Parse info lines
            if line.startswith('info'):
                # Check if it's a UCI info line (not a string message)
                if 'string' not in line and ('depth' in line or 'score' in line or 'pv' in line):
                    parts = line.split()
                    for i, part in enumerate(parts):
                        # Parse depth
                        if part == 'depth' and i+1 < len(parts):
                            info['depth'] = int(parts[i+1])
                        
                        # Parse selective depth
                        elif part == 'seldepth' and i+1 < len(parts):
                            info['seldepth'] = int(parts[i+1])
                        
                        # Parse score
                        elif part == 'score' and i+2 < len(parts):
                            if parts[i+1] == 'cp':  # Centipawns
                                info['score'] = int(parts[i+2]) / 100.0
                            elif parts[i+1] == 'mate':  # Mate in X
                                info['score'] = f"#{parts[i+2]}"
                        
                        # Parse nodes
                        elif part == 'nodes' and i+1 < len(parts):
                            info['nodes'] = int(parts[i+1])
                        
                        # Parse time
                        elif part == 'time' and i+1 < len(parts):
                            info['time'] = int(parts[i+1])
                        
                        # Parse nps (nodes per second)
                        elif part == 'nps' and i+1 < len(parts):
                            info['nps'] = int(parts[i+1])
                        
                        # Parse hashfull
                        elif part == 'hashfull' and i+1 < len(parts):
                            info['hashfull'] = int(parts[i+1])
                        
                        # Parse principal variation
                        elif part == 'pv' and i+1 < len(parts):
                            info['pv'] = parts[i+1:]
                
                # Parse WDL (win/draw/loss) statistics
                elif 'wdl' in line:
                    parts = line.split()
                    wdl_index = parts.index('wdl')
                    if wdl_index + 3 < len(parts):
                        info['wdl'] = {
                            'win': int(parts[wdl_index+1]) / 1000.0,
                            'draw': int(parts[wdl_index+2]) / 1000.0,
                            'loss': int(parts[wdl_index+3]) / 1000.0
                        }
            
            # If we found the best move
            if line.startswith('bestmove'):
                parts = line.split()
                if len(parts) >= 2:
                    best_move = parts[1]
                    return best_move, info
        
        raise TimeoutError(f"Timeout waiting for bestmove from engine. Got: {output}")
    
    def analyse(self, board, limit):
        """Analyse the position with the given limit"""
        # Set up the position
        fen_parts = board.fen().split()
        if len(fen_parts) < 6:
            # Add missing half move and full move numbers if needed
            fen = f"{board.fen()} 0 1"
        else:
            fen = board.fen()
            
        self._send_command(f"position fen {fen}")
        
        # Send the go command with appropriate time limits
        go_command = "go"
        if limit.time is not None:
            go_command += f" movetime {int(limit.time * 1000)}"
        elif limit.depth is not None:
            go_command += f" depth {limit.depth}"
        else:
            go_command += " depth 10"  # Default depth
            
        self._send_command(go_command)
        
        # Wait for the best move
        best_move_str, info = self._read_until_bestmove()
        
        # Convert best move string to Move object
        best_move = chess.Move.from_uci(best_move_str)
        
        # Add best move to PV if it's not already there
        pv = []
        if info['pv']:
            for move_str in info['pv']:
                try:
                    pv.append(chess.Move.from_uci(move_str))
                except ValueError:
                    # Skip invalid moves in PV
                    pass
        
        if not pv or pv[0] != best_move:
            pv = [best_move] + pv
            
        info['pv'] = pv
        
        # Add WDL info to score if available
        if info['wdl']:
            info['wdl_info'] = f" (W: {info['wdl']['win']:.1%}, D: {info['wdl']['draw']:.1%}, L: {info['wdl']['loss']:.1%})"
        
        return info
    
    def quit(self):
        """Quit the engine"""
        if self.process and self.process.poll() is None:
            self._send_command('quit')
            self.process.wait(timeout=5)
            
# root(index) route
@app.route('/')
def root():
    return render_template('bbc.html')

# make move API
@app.route('/make_move', methods=['POST'])
def make_move():
    # extract FEN string from HTTP POST request body
    pgn = request.form.get('pgn')
    
    # check if engine is enabled
    engine_enabled = request.form.get('engine_enabled', 'true').lower() == 'true'
    
    # Get engine choice (default to BBC)
    engine_choice = request.form.get('engine_choice', 'bbc')
    
    # read game moves from PGN
    game = chess.pgn.read_game(io.StringIO(pgn))    
    
    # init board
    board = game.board()
    
    # loop over moves in game
    for move in game.mainline_moves():
        # make move on chess board
        board.push(move)
    
    # probe opening book
    book_move = probe_book(pgn)
    if book_move:
        try:
            # Convert book move (in SAN) to UCI format
            move_obj = board.parse_san(book_move)
            uci_move = move_obj.uci()
            board.push(move_obj)
            
            return {
                'fen': board.fen(),
                'score': 'book move',
                'best_move': uci_move,
                'depth': '0',
                'pv': uci_move,
                'nodes': '0',
                'time': '0'
            }
        except Exception as e:
            print(f"Error parsing book move: {e}")
    
    # if engine is disabled, don't make a move
    if not engine_enabled:
        return {
            'fen': board.fen(),
            'best_move': '',
            'score': 'engine disabled',
            'depth': '0',
            'pv': '',
            'nodes': '0',
            'time': '0'
        }
    
    # Create appropriate engine based on choice
    if engine_choice == 'mcts':
        engine = PyTorchMCTSEngine()
    else:
        # Default to BBC engine
        engine_path = ENGINE_CHOICES.get(engine_choice, ENGINE_CHOICES['bbc'])
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    
    # extract fixed depth value
    fixed_depth = request.form.get('fixed_depth')

    # extract move time value
    move_time = request.form.get('move_time')
    
    # if move time is available
    if move_time != '0':
        if move_time == 'instant':
            try:
                # search for best move instantly
                info = engine.analyse(board, chess.engine.Limit(time=0.1))
            except Exception as e:
                print(f"Error analyzing with time limit 0.1: {e}")
                info = {}
        else:
            try:
                # search for best move with fixed move time
                info = engine.analyse(board, chess.engine.Limit(time=int(move_time)))
            except Exception as e:
                print(f"Error analyzing with time limit {move_time}: {e}")
                info = {}

    # if fixed depth is available
    elif fixed_depth != '0':
        try:
            # search for best move with fixed depth
            info = engine.analyse(board, chess.engine.Limit(depth=int(fixed_depth)))
        except Exception as e:
            print(f"Error analyzing with depth {fixed_depth}: {e}")
            info = {}
    else:
        # Default to depth 10
        try:
            info = engine.analyse(board, chess.engine.Limit(depth=10))
        except Exception as e:
            print(f"Error analyzing with default depth: {e}")
            info = {}
    
    # terminate engine process
    engine.quit()
    
    try:
        # extract best move from PV
        best_move = info['pv'][0]

        # update internal python chess board state
        board.push(best_move)
        
        # get best score
        if 'score' in info:
            if isinstance(info['score'], int) or isinstance(info['score'], float):
                score_value = -float(info['score'])
                # Format score with color indication similar to BBC format
                side_to_move = 'BLACK' if board.turn == chess.BLACK else 'WHITE'
                score = f"PovScore(Cp({int(score_value)}), {side_to_move})"
            else:
                score = str(info['score'])
                
                # inverse score
                if '+' in score:
                    score = score.replace('+', '-')
                
                elif '-' in score:
                    score = score.replace('-', '+')
        else:
            score = 0.0
            
        # Include WDL stats if available
        wdl_data = None
        if 'wdl' in info and info['wdl']:
            wdl_data = {
                'win': info['wdl']['win'],
                'draw': info['wdl']['draw'],
                'loss': info['wdl']['loss']
            }
          
        # Format PV in a more readable way
        formatted_pv = ""
        if 'pv' in info and info['pv']:
            temp_board = board.copy()
            temp_board.pop()  # Remove the best move we just applied
            for move in info.get('pv', [])[:6]:  # Limit to 6 moves for readability
                try:
                    if isinstance(move, str):
                        move_obj = chess.Move.from_uci(move)
                    else:
                        move_obj = move
                    san_move = temp_board.san(move_obj)
                    formatted_pv += san_move + " "
                    temp_board.push(move_obj)
                except Exception as e:
                    print(f"Error formatting PV: {e}")
                    break
        
        return {
            'fen': board.fen(),
            'best_move': str(best_move),
            'score': score,
            'depth': info.get('depth', 0),
            'pv': ' '.join([str(move) for move in info.get('pv', [])]),
            'formatted_pv': formatted_pv.strip(),
            'nodes': info.get('nodes', 0),
            'time': info.get('time', 0),
            'engine': engine_choice,
            'wdl': wdl_data
        }
    
    except Exception as e:
        print(f"Error processing engine response: {e}")
        return {
            'fen': board.fen(),
            'score': '#+1',
            'engine': engine_choice
        }

@app.route('/analytics')
def analytics():
    return render_template('stats.html')

@app.route('/analytics/api/post', methods=['POST'])
def post():
    response = Response('')
    response.headers['Access-Control-Allow-Origin'] = '*'

    stats = {
        'Date': request.form.get('date'),
        'Url': request.form.get('url'),
        'Agent':request.headers.get('User-Agent')
    }

    if request.headers.getlist("X-Forwarded-For"):
       stats['Ip'] = request.headers.getlist("X-Forwarded-For")[0]
    else:
       stats['Ip'] = request.remote_addr
    
    if request.headers.get('Origin'):
        stats['Origin'] = request.headers.get('Origin')
    else:
        stats['Origin'] = 'N/A'
    
    if request.headers.get('Referer'):
        stats['Referer'] = request.headers.get('Referer')
    else:
        stats['Referer'] = 'N/A'
    
    with open('stats.json', 'a') as f: f.write(json.dumps(stats, indent=2) + '\n\n')
    return response


@app.route('/analytics/api/get')
def get():
    stats = []
    
    with open('stats.json') as f:
        for entry in f.read().split('\n\n'):
            try: stats.append(json.loads(entry))
            except: pass
              
    return jsonify({'data': stats})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
