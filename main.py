import sys
from predictor import FootballMatchPredictor

if __name__ == "__main__":
    # Initialize and train the predictor
    predictor = FootballMatchPredictor('Premier League')
    predictor.train("/Datasets/season-2425.csv")

    print("\nEnter match-ups (format: HomeTeam vs AwayTeam). Type 'exit', 'close', or 'quit' to exit.")
    # Loop until user exits
    while True:
        try:
            user_in = input("Match: ").strip()
            # Exit commands
            if user_in.lower() in ('exit', 'close', 'quit'):
                print("Exiting.")
                sys.exit(0)

            # Expect format "HomeTeam vs AwayTeam"
            if 'vs' in user_in.lower():
                parts = user_in.split('vs', 1)
                home, away = parts[0].strip(), parts[1].strip()
            else:
                print("Invalid format. Use 'HomeTeam vs AwayTeam'.")
                continue

            # Perform prediction
            result = predictor.predict_match(home, away)
            print(f"Prediction -> {home} vs {away}: Home win: {result['home_win']:.2f}, "
                  f"Draw: {result['draw']:.2f}, Away win: {result['away_win']:.2f}")

        except KeyboardInterrupt:
            print("Keyboard interrupt received. Exiting.")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")