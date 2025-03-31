"""
UCI (Universal Chess Interface) protocol handler.
"""
import sys
import threading
import time
import chess

from src.engine import Engine

class UCIHandler:
    """Handles UCI protocol communication between the engine and a chess GUI."""
    
    def __init__(self, engine=None, name="PyTorch MCTS Chess Engine", author="AI Developer"):
        """Initialize the UCI handler.
        
        Args:
            engine: The Chess Engine instance to use. If None, create a default Engine.
            name: Engine name to report to UCI.
            author: Engine author to report to UCI.
        """
        self.engine = engine
        self.name = name
        self.author = author
        self.board = chess.Board()
        
        # Search state
        self.search_thread = None
        self.stop_search = False
        self.searching = False
        
        # Options
        self.options = {
            "Threads": 1,
            "Hash": 16,  # MB
            "Temperature": 0.0  # Move selection temperature
        }
    
    def uci_loop(self):
        """Main UCI command loop."""
        while True:
            if not sys.stdin.isatty():
                # GUI mode - reads input from stdin
                try:
                    line = sys.stdin.readline().strip()
                    if not line:
                        continue
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error reading input: {e}")
                    break
            else:
                # Interactive mode for testing
                try:
                    line = input()
                except KeyboardInterrupt:
                    break
                except EOFError:
                    break
            
            if not line:
                continue
                
            if line == "quit":
                break
                
            try:
                self.parse_command(line)
            except Exception as e:
                print(f"info string Error processing command: {e}")
    
    def parse_command(self, line):
        """Parse a UCI command."""
        parts = line.split()
        if not parts:
            return
            
        command = parts[0]
        
        if command == "uci":
            self.cmd_uci()
        elif command == "isready":
            self.cmd_isready()
        elif command == "setoption":
            self.cmd_setoption(parts[1:])
        elif command == "ucinewgame":
            self.cmd_ucinewgame()
        elif command == "position":
            self.cmd_position(parts[1:])
        elif command == "go":
            self.cmd_go(parts[1:])
        elif command == "stop":
            self.cmd_stop()
        elif command == "quit":
            # Handled in the main loop
            pass
        else:
            self.send_response(f"info string Unknown command: {command}")
    
    def cmd_uci(self):
        """Handle the 'uci' command."""
        self.send_response(f"id name {self.name}")
        self.send_response(f"id author {self.author}")
        
        # Report options
        self.send_response("option name Threads type spin default 1 min 1 max 8")
        self.send_response("option name Hash type spin default 16 min 1 max 1024")
        self.send_response("option name Temperature type spin default 0 min 0 max 100")
        
        self.send_response("uciok")
    
    def cmd_isready(self):
        """Handle the 'isready' command."""
        # Initialize engine if needed
        if self.engine is None:
            self.engine = Engine()
        self.send_response("readyok")
    
    def cmd_setoption(self, args):
        """Handle the 'setoption' command."""
        if len(args) < 2 or args[0] != "name":
            return
            
        name_parts = []
        value_parts = []
        
        i = 1
        while i < len(args) and args[i] != "value":
            name_parts.append(args[i])
            i += 1
            
        if i < len(args) and args[i] == "value":
            i += 1
            while i < len(args):
                value_parts.append(args[i])
                i += 1
        
        name = " ".join(name_parts)
        value = " ".join(value_parts) if value_parts else None
        
        self.set_option(name, value)
    
    def cmd_ucinewgame(self):
        """Handle the 'ucinewgame' command."""
        # Reset the board to the starting position
        self.board = chess.Board()
        # Clear any persistent state in the engine
        # (Not needed for this implementation as we create a new search tree each time)
    
    def cmd_position(self, args):
        """Handle the 'position' command."""
        if not args:
            return
            
        # Parse the position command
        i = 0
        if args[0] == "startpos":
            # Make sure to create a brand new board to reset all state
            self.board = chess.Board()
            i = 1
        elif args[0] == "fen":
            # Find where the FEN ends and moves start
            fen_parts = []
            i = 1
            while i < len(args) and args[i] != "moves":
                fen_parts.append(args[i])
                i += 1
                
            # Set up the position with a fresh board instance
            try:
                fen = " ".join(fen_parts)
                self.board = chess.Board(fen)
            except ValueError as e:
                self.send_response(f"info string Invalid FEN: {e}")
                return
        else:
            # Invalid command
            self.send_response(f"info string Invalid position command: {args[0]}")
            return
            
        # Apply any moves
        if i < len(args) and args[i] == "moves":
            i += 1
            # Debug the moves list to help diagnose issues
            move_list = args[i:]
            self.send_response(f"info string Applying moves: {' '.join(move_list)}")
            
            while i < len(args):
                try:
                    move = chess.Move.from_uci(args[i])
                    if move in self.board.legal_moves:
                        self.board.push(move)
                    else:
                        self.send_response(f"info string Illegal move: {args[i]}")
                        return
                except ValueError:
                    self.send_response(f"info string Invalid move: {args[i]}")
                    return
                i += 1
                
        # Debug the final position
        self.send_response(f"info string Final position FEN: {self.board.fen()}")
    
    def cmd_go(self, args):
        """Handle the 'go' command."""
        # Stop any ongoing search
        self.cmd_stop()
        
        # Parse parameters
        time_control = {
            'wtime': None,
            'btime': None,
            'winc': None,
            'binc': None,
            'movestogo': None,
            'depth': None,
            'nodes': None,
            'movetime': None,
            'infinite': False,
        }
        
        i = 0
        while i < len(args):
            if args[i] in time_control:
                if args[i] == "infinite":
                    time_control['infinite'] = True
                    i += 1
                else:
                    if i + 1 < len(args):
                        try:
                            time_control[args[i]] = int(args[i + 1])
                        except ValueError:
                            self.send_response(f"info string Invalid value for {args[i]}: {args[i + 1]}")
                        i += 2
                    else:
                        i += 1
            else:
                # Skip unknown parameters
                i += 1
        
        # Calculate time to use for the search
        movetime = self._calculate_time(time_control)
        
        # Start the search in a separate thread
        self.stop_search = False
        self.searching = True
        self.search_thread = threading.Thread(
            target=self._search_and_report,
            args=(movetime, time_control['depth'], time_control['nodes']),
            daemon=True
        )
        self.search_thread.start()
    
    def cmd_stop(self):
        """Handle the 'stop' command."""
        if self.searching:
            self.stop_search = True
            if self.search_thread and self.search_thread.is_alive():
                self.search_thread.join(timeout=1.0)
            self.searching = False
    
    def set_option(self, name, value):
        """Handle setting an option."""
        if name in self.options:
            # Convert value to the appropriate type
            try:
                if name == "Temperature":
                    # Temperature is stored as [0, 100] in UCI but used as [0, 1] internally
                    self.options[name] = float(value) / 100.0
                else:
                    self.options[name] = int(value)
                self.send_response(f"info string Option {name} set to {value}")
            except ValueError:
                self.send_response(f"info string Invalid value for option {name}: {value}")
        else:
            self.send_response(f"info string Unknown option: {name}")
    
    def _calculate_time(self, time_control):
        """Calculate how much time to use for the current search."""
        # If movetime is specified, use that directly
        if time_control['movetime'] is not None:
            return time_control['movetime']
            
        # If infinite, use a very large value
        if time_control['infinite']:
            return 2147483647  # Approx. 24 days, effectively infinite
            
        # Use time control if available
        if self.board.turn == chess.WHITE and time_control['wtime'] is not None:
            time_left = time_control['wtime']
            inc = time_control['winc'] or 0
        elif self.board.turn == chess.BLACK and time_control['btime'] is not None:
            time_left = time_control['btime']
            inc = time_control['binc'] or 0
        else:
            # Default to a reasonable time if not specified
            return 1000  # 1 second
            
        # Simple time management: use a fraction of the remaining time
        movestogo = time_control['movestogo'] or 30  # Default assumption
        time_per_move = (time_left / movestogo) + (inc * 0.75)
        
        # Ensure we don't use too much time
        time_per_move = min(time_per_move, time_left / 10)
        
        # Ensure we use at least a minimal amount of time
        time_per_move = max(time_per_move, 100)  # At least 100ms
        
        return int(time_per_move)
    
    def _search_and_report(self, movetime, depth, nodes):
        """Execute the search and report the best move."""
        try:
            # Prepare parameters
            if depth is not None:
                self.send_response(f"info string Searching to depth {depth}")
            elif nodes is not None:
                self.send_response(f"info string Searching {nodes} nodes")
            else:
                self.send_response(f"info string Searching for {movetime}ms")
            
            # Start search timer
            start_time = time.time()
            
            # Execute the search
            temp = self.options.get("Temperature", 0.0)
            best_move = self.engine.search(
                self.board,
                time_limit_ms=movetime,
                depth_limit=depth,
                nodes_limit=nodes,
                temperature=temp
            )
            
            # Search complete
            search_time = int((time.time() - start_time) * 1000)
            
            if best_move is None:
                # No legal moves
                self.send_response("info string No legal moves")
                self.send_response("bestmove 0000")
            else:
                # Report the best move
                self.send_response(f"info time {search_time}")
                self.send_response(f"bestmove {best_move.uci()}")
        
        except Exception as e:
            self.send_response(f"info string Error in search: {e}")
            self.send_response("bestmove a1a1")  # Send a dummy move
        
        finally:
            self.searching = False
    
    def send_response(self, message):
        """Send a response to the GUI."""
        print(message, flush=True) 