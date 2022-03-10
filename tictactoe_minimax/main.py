from Board import Board
from State import State
from Player import Human, AI
from random import choice


def main() -> int:
    board = Board()
    state = State(board)
    board.print_board()
    player1 = Human("X")
    player2 = AI("O")
    # player2 = Human("O")
    turn = choice([player1, player2])
    depth = 10
    # turn = player1
    print(f"Zaczyna gracz {turn.mark}")
    while state.result(player2.mark, player1.mark) is None:
        if turn == player1:
            player1.move(board)
            turn = player2
        else:
            player2.move(board, player1.mark, depth)
            # player2.move(board)
            turn = player1
        board.print_board()

    if state.result(player2.mark, player1.mark) == 100:
        print(f"Gracz: {player2.mark} wygrał!")
    if state.result(player2.mark, player1.mark) == -100:
        print(f"Gracz: {player1.mark} wygrał!")
    if state.result(player2.mark, player1.mark) == 0:
        print("Remis!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
