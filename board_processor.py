import numpy as np
from typing import List, Optional, Tuple


class BoardProcessor:
    """
    Handles the game-board state representation, move encoding/decoding, and ASCII display logic.
    """

    def __init__(self, rows: int = 6, cols: int = 7):
        self.rows = rows
        self.cols = cols
        # Base-N numerals for encoding/decoding
        self.numerals = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        # Internal state
        self.move_sequence: List[int] = []
        self.state_list = self.reset_state_list()
        self.board_history: List[
            np.ndarray
        ] = []  # list of board matrices after each move

    def reset_state_list(self) -> List[List[int]]:
        """
        Initialize an empty state list: one list per column.
        """
        self.state_list = [[] for _ in range(self.cols)]
        self.board_history = []
        return self.state_list

    def generate_state_list(
        self, move_sequence: List[int], initial_move: int = 1
    ) -> Tuple[int, List[List[int]]]:
        """
        Apply a sequence of moves to an empty board, alternating players,
        storing intermediate board states in history.

        Returns:
            (last_valid_index, state_list)
        """
        if not isinstance(move_sequence, list):
            raise TypeError("move_sequence must be a list of column indices")
        self.reset_state_list()
        self.move_sequence = move_sequence
        self.board_history = []
        player = initial_move
        last_idx = -1
        for cnt, col in enumerate(move_sequence):
            if not isinstance(col, int):
                raise TypeError(f"Move at position {cnt} is not an integer: {col}")
            if col < 0 or col >= self.cols:
                raise ValueError(f"Move at position {cnt} out of bounds: {col}")
            if len(self.state_list[col]) >= self.rows:
                return last_idx, self.state_list
            # place piece
            self.state_list[col].append(player)
            # record board after move
            board = self._build_board_matrix(self.state_list)
            self.board_history.append(board)
            player *= -1
            last_idx = cnt
        return last_idx, self.state_list

    def _build_board_matrix(self, state_list: List[List[int]]) -> np.ndarray:
        """
        Convert state_list to a full board matrix (rows x cols).
        """
        board = np.zeros((self.rows, self.cols), dtype=int)
        for col, col_list in enumerate(state_list):
            for row, val in enumerate(col_list):
                board[row, col] = val
        return board

    def display_board(self, index: Optional[int] = None) -> None:
        """
        Print the board at a given history index (default last) using:
          'X' for player 1,
          'O' for player -1,
          '.' for empty.
        """
        if not self.board_history:
            print("[Board is empty]")
            return
        if index is None:
            board = self.board_history[-1]
        else:
            if index < 0 or index >= len(self.board_history):
                raise IndexError(f"History index out of range: {index}")
            board = self.board_history[index]
        symbols = {1: "X", -1: "O", 0: "."}
        # Print top row first
        for r in range(self.rows - 1, -1, -1):
            row_str = " ".join(symbols.get(board[r, c], "?") for c in range(self.cols))
            print(row_str)

    def base_n(self, num: int, base: int) -> str:
        """
        Convert a non-negative integer to a base-N string using instance numerals.
        """
        if not isinstance(num, int) or num < 0:
            raise ValueError("num must be a non-negative integer")
        if base < 2 or base > len(self.numerals):
            raise ValueError(
                f"base must be between 2 and {len(self.numerals)} inclusive"
            )
        if num == 0:
            return self.numerals[0]
        digits = []
        n = num
        while n > 0:
            n, rem = divmod(n, base)
            digits.append(self.numerals[rem])
        return "".join(reversed(digits))

    def moves_code(self, step: Optional[int] = None) -> str:
        """
        Encode self.move_sequence up to a given step into a compact base-62 string.

        Args:
            step: Optional index (0-based) to limit moves (inclusive). If None, uses full sequence.

        Returns:
            Base-62 encoded string of moves up to step.
        """
        moves = self.move_sequence
        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer index")
            if step < 0 or step >= len(moves):
                raise ValueError(f"step out of range: {step}")
            moves = moves[: step + 1]
        # validate moves list
        if not isinstance(moves, list):
            raise TypeError("move_sequence must be a list of integers")
        for cnt, m in enumerate(moves):
            if not isinstance(m, int):
                raise TypeError(f"Move at index {cnt} is not an integer: {m}")
            if m < 0 or m >= self.cols:
                raise ValueError(f"Move at index {cnt} out of range: {m}")
        s = "1" + "".join(str(m) for m in moves)
        total = int(s, 7)
        return self.base_n(total, 62)

    def decode_moves_code(self, code: str) -> List[int]:
        if not isinstance(code, str) or not code:
            raise TypeError("code must be a non-empty string")
        char_to_val = {ch: idx for idx, ch in enumerate(self.numerals)}
        total = 0
        for ch in code:
            if ch not in char_to_val:
                raise ValueError(f"Invalid character in code: {ch}")
            total = total * 62 + char_to_val[ch]
        # to base-7
        s = ""
        temp = total
        while temp > 0:
            temp, rem = divmod(temp, 7)
            s = str(rem) + s
        if not s or s[0] != "1":
            raise ValueError("Invalid moves code: missing prefix '1'")
        moves = [int(ch) for ch in s[1:]]
        # regenerate history
        self.generate_state_list(moves)
        return moves
