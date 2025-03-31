"""
Board representation utilities - conversion between chess.Board and input planes for the neural network.
"""
import chess
import numpy as np
import torch

# Constants for board representation
# Piece planes: first 6 are white pieces, next 6 are black pieces
# Order: Pawn, Knight, Bishop, Rook, Queen, King
PIECE_TO_PLANE_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}

# Maximum number of UCI encoded moves - for defining policy vector size
MAX_UCI_MOVES = 4672  # 64 origins × 73 destinations = 4672 possible moves

def move_to_policy_index(move, board=None):
    """
    Convert a chess.Move to a unique index in the policy vector.
    
    Args:
        move: A chess.Move object.
        board: Optional chess.Board object (used for promotions).
        
    Returns:
        int: The index for this move in the policy vector.
    """
    # For simplicity, map based on origin square index (0-63) and destination square index (0-63)
    # This doesn't fully account for promotions but is a starting point
    from_square = move.from_square
    to_square = move.to_square
    
    # Basic index calculation: 64 possible from squares * 64 possible to squares
    index = from_square * 64 + to_square
    
    # Handle promotions by adding an offset
    if move.promotion:
        # Promotion piece type is in range 2-5 (knight, bishop, rook, queen)
        # Subtract 2 to get 0-3 range, then add it as an offset
        promo_offset = (move.promotion - 2) * (64 * 64)
        index += promo_offset
    
    return index

def policy_index_to_move(index, board):
    """
    Convert a policy vector index back to a chess.Move.
    
    Args:
        index: Integer index in the policy vector.
        board: A chess.Board object to validate the move.
        
    Returns:
        chess.Move: The move corresponding to the index.
    """
    # Determine if this is a promotion move
    promotion = None
    if index >= 64 * 64:
        # Extract the promotion piece type
        promo_number = index // (64 * 64)
        promotion = promo_number + 2  # Add 2 to get back to the chess piece type
        index = index % (64 * 64)  # Get the base move index
    
    # Extract from and to squares
    from_square = index // 64
    to_square = index % 64
    
    # Create the move
    move = chess.Move(from_square=from_square, to_square=to_square, promotion=promotion)
    
    # Validate the move is legal on the given board
    if board and move not in board.legal_moves:
        raise ValueError(f"Generated move {move} is not legal on the current board")
        
    return move

def board_to_input_planes(board, history_length=1):
    """
    Convert a chess.Board to a set of input planes for the neural network.
    
    Args:
        board: A chess.Board object.
        history_length: Number of past positions to include (including current).
        
    Returns:
        np.ndarray: Input planes with shape (C, 8, 8) where C is the number of channels.
    """
    # Calculate total number of planes (features per position × history length)
    num_features_per_position = 19  # 12 piece planes + 4 castling + 1 side-to-move + 1 en-passant + 1 fifty-move
    total_planes = num_features_per_position * history_length
    
    # Initialize input planes with zeros
    planes = np.zeros((total_planes, 8, 8), dtype=np.float32)
    
    # Process the current board state first
    _fill_board_planes(board, planes, offset=0)
    
    # If history is requested, try to reconstruct previous states
    if history_length > 1 and board.move_stack:
        # Make a copy of the board to avoid modifying the original
        temp_board = board.copy()
        
        # Process each historical position
        for h in range(1, history_length):
            if not temp_board.move_stack:
                # No more moves to undo, leave the remaining planes as zeros
                break
                
            # Undo the last move
            temp_board.pop()
            
            # Fill planes for this historical position
            _fill_board_planes(temp_board, planes, offset=h * num_features_per_position)
    
    return planes

def _fill_board_planes(board, planes, offset=0):
    """
    Fill the input planes for a single board position.
    
    Args:
        board: A chess.Board object.
        planes: The np.ndarray to fill.
        offset: The starting index in the planes to fill.
    """
    # 1-12: Piece planes (6 white, 6 black)
    for square in chess.SQUARES:
        # Get the piece at this square (if any)
        piece = board.piece_at(square)
        if piece:
            # Calculate indices
            row, col = square_to_row_col(square)
            color_offset = 0 if piece.color == chess.WHITE else 6
            plane_idx = offset + PIECE_TO_PLANE_INDEX[piece.piece_type] + color_offset
            
            # Set the value to 1 in the appropriate plane
            planes[plane_idx, row, col] = 1
    
    # 13: Side to move - fill with 1s if white to move
    if board.turn == chess.WHITE:
        planes[offset + 12] = 1
    
    # 14-17: Castling rights
    castling_rights = [
        board.has_kingside_castling_rights(chess.WHITE),    # White kingside
        board.has_queenside_castling_rights(chess.WHITE),   # White queenside
        board.has_kingside_castling_rights(chess.BLACK),    # Black kingside
        board.has_queenside_castling_rights(chess.BLACK)    # Black queenside
    ]
    for i, has_right in enumerate(castling_rights):
        if has_right:
            planes[offset + 13 + i] = 1
    
    # 18: En passant square
    if board.ep_square is not None:
        row, col = square_to_row_col(board.ep_square)
        planes[offset + 17, row, col] = 1
    
    # 19: Fifty-move counter (normalized to 0-1 range)
    planes[offset + 18] = min(1.0, board.halfmove_clock / 100.0)

def square_to_row_col(square):
    """
    Convert a chess.square (0-63) to row, col coordinates.
    
    Args:
        square: An integer square index (0-63).
        
    Returns:
        tuple: A (row, col) coordinate pair where 0,0 is the bottom-left corner (a1).
    """
    # Chess squares are 0=a1, 7=h1, 56=a8, 63=h8
    row = 7 - (square // 8)  # Rows are 0-7 bottom to top
    col = square % 8         # Columns are 0-7 left to right
    return row, col

def input_planes_to_tensor(planes):
    """
    Convert numpy input planes to PyTorch tensor.
    
    Args:
        planes: np.ndarray of input planes.
        
    Returns:
        torch.Tensor: The input tensor for the model.
    """
    return torch.from_numpy(planes).float() 