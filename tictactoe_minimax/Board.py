from typing import ValuesView, KeysView


class Board:
    def __init__(self) -> None:
        self.board = self.empty_board()

    def __getitem__(self, item: tuple) -> str:
        return self.board[item]

    def keys(self) -> KeysView:
        return self.board.keys()

    def values(self) -> ValuesView:
        return self.board.values()

    def full(self) -> bool:
        return list(self.values()).count(" ") == 0

    @staticmethod
    def empty_board() -> dict:
        board = {}
        for x in range(3):
            for y in range(3):
                board[(x, y)] = " "
        return board

    def print_board(self) -> None:
        print("\n")
        print(" " * 2 + "1" + " " * 5 + "2" + " " * 5 + "3")
        for coord in sorted(self.board.keys()):
            x, y = coord
            if y == 0 and x != 0:  # new row
                print("\u2500" * 5 + "\u253C" + "\u2500" * 5 + "\u253C" + "\u2500" * 5)
            val = self.board[coord]
            if val != " ":
                if y != 2:
                    print(f"  {val}  \u2502", end="")
                else:
                    print(f"  {val}  {x + 1}", end="\n")
            else:
                if y != 2:
                    print("     \u2502", end="")
                else:
                    print(" " * 5 + str(x + 1))
        print("\n")
