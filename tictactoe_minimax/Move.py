class Move:
    def __init__(self, x: int, y: int, board, player_m: str):
        self.x = x - 1
        self.y = y - 1
        self.board = board
        self.player_m = player_m

    def make_move(self):
        self.board.board[(self.x, self.y)] = self.player_m

    def is_valid(self) -> bool:
        return self.board.board[(self.x, self.y)] == " "
