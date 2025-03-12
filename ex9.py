def main():
    game_size = 9
    strike = 6
    _network = GameNetwork(size=game_size)

    # Create a PUCT player.
    puct_player = PUCTPlayer(neural_network=_network, cpuct=1)

    # create a MCTS player for comparison
    comp_player1 = MCTSPlayer(maximizer_player=-1)
    # comp_player1 = None
    agent_rating = 20
    opponent_rating = 50
    rater = Rating(agent_rating=agent_rating, opponent_rating=opponent_rating)
    results = []

    for _ in range(30):
        print(f"game {_ + 1}")
        _game = GomokuGame(size=game_size, strike=strike)
        while _game.ONGOING:
            chosen_move = puct_player.choose_move(
                game_instance=_game, iterations=agent_rating
            )
            print("PUCT (Black) selected move:", chosen_move)
            _game.make_move(chosen_move)

            if not _game.ONGOING:
                break

            if comp_player1:
                chosen_move, _ = comp_player1.choose_move(
                    game=_game, iterations=opponent_rating
                )
                print("MCTS (White) selected move:", chosen_move)
            else:
                chosen_move = tuple(
                    map(int, input("Enter your move as 'row col': ").split())
                )
            _game.make_move(chosen_move)
            print(_game)

        rating_agent, rating_opponent = rater.update_ratings(_game.winner, k=32)
        if _game.winner == _game.PLAYER_BLACK:
            print("\nBlack (Player 1) wins!")
        elif _game.winner == _game.PLAYER_WHITE:
            print("\nWhite (Player 2) wins!")
        else:
            print("\nIt's a draw!")

        results.append(_game.winner)
        print("New rating for agent: {:.2f}".format(rating_agent))
        print("New rating for opponent: {:.2f}".format(rating_opponent))

    print(f"Black {results.count(1)}")
    print(f"White {results.count(-1)}")
    print("Draw", results.count(0))
