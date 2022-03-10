from __future__ import annotations


class State:
    VALUES = {
        (0, 0): 3,
        (0, 1): 2,
        (0, 2): 3,
        (1, 0): 2,
        (1, 1): 4,
        (1, 2): 2,
        (2, 0): 3,
        (2, 1): 2,
        (2, 2): 3,
    }

    def __init__(self, board) -> None:
        self.board = board

    def has_won(self, player_mark: str) -> bool:
        diagonal_1 = [self.board[x, y] for x, y in sorted(self.board.keys()) if x is y]
        diagonal_2 = [
            self.board[x, y] for x, y in sorted(self.board.keys()) if x + y == 2
        ]
        for n in range(3):
            rows = [self.board[x, y] for x, y in sorted(self.board.keys()) if n is x]
            cols = [self.board[x, y] for x, y in sorted(self.board.keys()) if n is y]
            if (
                rows.count(player_mark) == 3
                or cols.count(player_mark) == 3
                or diagonal_1.count(player_mark) == 3
                or diagonal_2.count(player_mark) == 3
            ):
                return True
        return False

    def result(self, AI_mark: str, human_mark: str) -> int | None:
        if self.has_won(human_mark):
            return -100
        if self.has_won(AI_mark):
            return 100
        if self.board.full():
            return 0
        return None

    def evaluate_position(self, AI_m: str, human_m: str) -> int:
        evaluation = 0
        if self.result(AI_m, human_m) is None:
            for key in self.board.keys():
                if self.board.board[key] == AI_m:
                    evaluation += self.VALUES[key]
                if self.board.board[key] == human_m:
                    evaluation -= self.VALUES[key]
            return evaluation
        return self.result(AI_m, human_m)
