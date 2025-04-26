import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict
from feature_engineering import prepare_features, update_team_stats, get_result

class FootballMatchPredictor:
    def __init__(self, league_name: str, model_dir: str = "models"):
        """
        Initialize predictor; attempts to load persisted model from disk.
        """
        self.league_name = league_name
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_columns = [
            'HomeTeam_goals_scored_avg', 'HomeTeam_goals_conceded_avg',
            'AwayTeam_goals_scored_avg', 'AwayTeam_goals_conceded_avg',
            'HomeTeam_form', 'AwayTeam_form',
            'HomeTeam_points', 'AwayTeam_points'
        ]
        self.team_stats: Dict = {}
        self.model_dir = model_dir
        self.model_path = os.path.join(self.model_dir, f"{league_name.replace(' ', '_').lower()}_model.pkl")
        # create models directory
        os.makedirs(self.model_dir, exist_ok=True)
        # try loading existing model
        if os.path.exists(self.model_path):
            self._load_model()

    def train(self, matches_input) -> None:
        """
        Train the prediction model and persist it.
        """
        # load DataFrame
        import pandas as pd
        matches_df = pd.read_csv(matches_input) if isinstance(matches_input, str) else matches_input

        # feature generation
        features_df = prepare_features(matches_df, self.team_stats)
        X = features_df[self.feature_columns]
        y = features_df['result']

        # split, scale, train
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)
        print(f"Train acc: {self.model.score(X_train_scaled, y_train):.3f}")
        print(f"Test  acc: {self.model.score(X_test_scaled, y_test):.3f}")

        # persist
        self._save_model()

    def predict_match(self, HomeTeam: str, AwayTeam: str) -> Dict:
        """
        Predict outcome probabilities for a given fixture.
        """
        if not self.team_stats:
            raise ValueError("Model not trained or loaded. Run training first.")

        # temporarily use feature prep on a single row
        import pandas as pd
        row = pd.DataFrame([{ 'Date': None, 'HomeTeam': HomeTeam, 'AwayTeam': AwayTeam, 'FTHG': None, 'FTAG': None }])
        features = prepare_features(row, self.team_stats)
        vals = [features[col].iloc[0] for col in self.feature_columns]
        scaled = self.scaler.transform([vals])
        home_p, draw_p, away_p = self.model.predict_proba(scaled)[0]
        return {'home_win': home_p, 'draw': draw_p, 'away_win': away_p}

    def _save_model(self) -> None:
        """Serialize model, scaler, and team_stats to disk."""
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'team_stats': self.team_stats
        }
        joblib.dump(data, self.model_path)

    def _load_model(self) -> None:
        """Load persisted model, scaler, and team_stats from disk."""
        data = joblib.load(self.model_path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.team_stats = data['team_stats']
        print(f"Loaded model from {self.model_path}")