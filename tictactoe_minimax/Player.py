import math
import re
from State import State
from Board import Board
from Move import Move


class Human:
    def __init__(self, mark: str) -> None:
        self.mark = mark

    def move(self, board):
        is_valid = False
        while not is_valid:
            p_input = input(f"Ruch gracza {self.mark}: ")
            if re.match(r"^[123]\s*,\s*[123]$", p_input):
                x = int(p_input.split(",")[0])
                y = int(p_input.split(",")[1])
                move = Move(x, y, board, self.mark)
                if move.is_valid():
                    move.make_move()
                    is_valid = True
                else:
                    print("To pole jest zajęte, wybierz inne")
            else:
                print(
                    "Zły format wejścia. Wprowadź dane w formacie -> x, y w przedziale (1,3)"
                )


class AI:
    def __init__(self, mark: str) -> None:
        self.mark = mark

    def move(self, board, opponent_m: str, depth: int) -> None:
        state = State(board)
        best_score = -math.inf
        best_move = ()
        for key in board.board.keys():
            if board.board[key] == " ":
                board.board[key] = self.mark
                score = self.minimax(board, state, False, opponent_m, depth)
                if state.result(self.mark, opponent_m) == 100:
                    best_move = key
                    break
                board.board[key] = " "
                if score > best_score:
                    best_score = score
                    best_move = key

        move = Move(best_move[0] + 1, best_move[1] + 1, board, self.mark)
        state.evaluate_position(self.mark, opponent_m)
        move.make_move()

    def minimax(
        self,
        board: Board,
        state: State,
        is_maximazing: bool,
        opponent_m: str,
        depth: int,
    ) -> int:
        if depth == 0 or state.result(self.mark, opponent_m) is not None:
            return state.evaluate_position(self.mark, opponent_m)
        if is_maximazing:
            best_score = -math.inf
            for key in board.keys():
                if board.board[key] == " ":
                    board.board[key] = self.mark
                    best_score = max(
                        best_score,
                        self.minimax(
                            board, state, not is_maximazing, opponent_m, depth - 1
                        ),
                    )
                    board.board[key] = " "
            return best_score
        else:
            best_score = math.inf
            for key in board.keys():
                if board.board[key] == " ":
                    board.board[key] = opponent_m
                    best_score = min(
                        best_score,
                        self.minimax(
                            board, state, not is_maximazing, opponent_m, depth - 1
                        ),
                    )
                    board.board[key] = " "
            return best_score
