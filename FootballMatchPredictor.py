import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Tuple, Optional

class FootballMatchPredictor:
    def __init__(self, league_name: str):
        """
        Initialize the predictor for a specific league
        
        Args:
            league_name (str): Name of the league (e.g., 'Premier League', 'La Liga')
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
        
    def prepare_features(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features for each match and store team statistics as a class attribute.
        """
        self.team_stats = {}
        
        teams = set(matches_df['HomeTeam'].unique()) | set(matches_df['AwayTeam'].unique())
        for team in teams:
            self.team_stats[team] = {
                'goals_scored': [], 'goals_conceded': [],
                'last_5_results': [], 'points': 0
            }
            
        features = []
        matches_df = matches_df.sort_values('Date')
        
        for _, match in matches_df.iterrows():
            HomeTeam = match['HomeTeam']
            AwayTeam = match['AwayTeam']
            home_goals = match['FTHG']
            away_goals = match['FTAG']
            
            home_stats = {
                'goals_scored_avg': np.mean(self.team_stats[HomeTeam]['goals_scored'][-5:]) if self.team_stats[HomeTeam]['goals_scored'] else 0,
                'goals_conceded_avg': np.mean(self.team_stats[HomeTeam]['goals_conceded'][-5:]) if self.team_stats[HomeTeam]['goals_conceded'] else 0,
                'form': np.mean(self.team_stats[HomeTeam]['last_5_results'][-5:]) if self.team_stats[HomeTeam]['last_5_results'] else 0,
                'points': self.team_stats[HomeTeam]['points']
            }
            
            away_stats = {
                'goals_scored_avg': np.mean(self.team_stats[AwayTeam]['goals_scored'][-5:]) if self.team_stats[AwayTeam]['goals_scored'] else 0,
                'goals_conceded_avg': np.mean(self.team_stats[AwayTeam]['goals_conceded'][-5:]) if self.team_stats[AwayTeam]['goals_conceded'] else 0,
                'form': np.mean(self.team_stats[AwayTeam]['last_5_results'][-5:]) if self.team_stats[AwayTeam]['last_5_results'] else 0,
                'points': self.team_stats[AwayTeam]['points']
            }
            
            features.append({
                'HomeTeam_goals_scored_avg': home_stats['goals_scored_avg'],
                'HomeTeam_goals_conceded_avg': home_stats['goals_conceded_avg'],
                'AwayTeam_goals_scored_avg': away_stats['goals_scored_avg'],
                'AwayTeam_goals_conceded_avg': away_stats['goals_conceded_avg'],
                'HomeTeam_form': home_stats['form'],
                'AwayTeam_form': away_stats['form'],
                'HomeTeam_points': home_stats['points'],
                'AwayTeam_points': away_stats['points'],
                'result': self._get_result(home_goals, away_goals)
            })
            
            self._update_team_stats(self.team_stats, HomeTeam, AwayTeam, home_goals, away_goals)
            
        return pd.DataFrame(features)
    
    def _update_team_stats(self, team_stats: Dict, HomeTeam: str, AwayTeam: str, 
                          home_goals: int, away_goals: int) -> None:
        """Update team statistics after a match"""
        # Update goals
        team_stats[HomeTeam]['goals_scored'].append(home_goals)
        team_stats[HomeTeam]['goals_conceded'].append(away_goals)
        team_stats[AwayTeam]['goals_scored'].append(away_goals)
        team_stats[AwayTeam]['goals_conceded'].append(home_goals)
        
        # Update form (1 for win, 0.5 for draw, 0 for loss)
        if home_goals > away_goals:
            team_stats[HomeTeam]['last_5_results'].append(1)
            team_stats[AwayTeam]['last_5_results'].append(0)
            team_stats[HomeTeam]['points'] += 3
        elif home_goals < away_goals:
            team_stats[HomeTeam]['last_5_results'].append(0)
            team_stats[AwayTeam]['last_5_results'].append(1)
            team_stats[AwayTeam]['points'] += 3
        else:
            team_stats[HomeTeam]['last_5_results'].append(0.5)
            team_stats[AwayTeam]['last_5_results'].append(0.5)
            team_stats[HomeTeam]['points'] += 1
            team_stats[AwayTeam]['points'] += 1
    
    def _get_result(self, home_goals: int, away_goals: int) -> int:
        """Convert goals to result category (0: Home win, 1: Draw, 2: Away win)"""
        if home_goals > away_goals:
            return 0
        elif home_goals < away_goals:
            return 2
        return 1
        
    def train(self, matches_input) -> None:
        """
        Train the prediction model.
        
        Args:
            matches_input: Either a file path (str) to a CSV file or a DataFrame with historical match data.
        """
        # Load dataset if input is a file path
        if isinstance(matches_input, str):
            matches_df = pd.read_csv(matches_input)
        elif isinstance(matches_input, pd.DataFrame):
            matches_df = matches_input
        else:
            raise ValueError("Input must be a file path (str) or a pandas DataFrame.")
        
        # Prepare features
        features_df = self.prepare_features(matches_df)
        
        # Correct feature columns to match prepare_features output
        self.feature_columns = [
            'HomeTeam_goals_scored_avg', 'HomeTeam_goals_conceded_avg',
            'AwayTeam_goals_scored_avg', 'AwayTeam_goals_conceded_avg',
            'HomeTeam_form', 'AwayTeam_form',
            'HomeTeam_points', 'AwayTeam_points'
        ]
        
        # Ensure the result column is present
        X = features_df[self.feature_columns]
        y = features_df['result']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Print model accuracy
        train_accuracy = self.model.score(X_train_scaled, y_train)
        test_accuracy = self.model.score(X_test_scaled, y_test)
        print(f"Training accuracy: {train_accuracy:.3f}")
        print(f"Test accuracy: {test_accuracy:.3f}")

    
    def predict_match(self, HomeTeam: str, AwayTeam: str) -> Dict:
        """
        Predict the outcome of a match using stored team statistics.
        """
        if not self.team_stats:
            raise ValueError("Team statistics are not available. Train the model first.")
        
        home_stats = {
            'goals_scored_avg': np.mean(self.team_stats[HomeTeam]['goals_scored'][-5:]),
            'goals_conceded_avg': np.mean(self.team_stats[HomeTeam]['goals_conceded'][-5:]),
            'form': np.mean(self.team_stats[HomeTeam]['last_5_results'][-5:]),
            'points': self.team_stats[HomeTeam]['points']
        }
        
        away_stats = {
            'goals_scored_avg': np.mean(self.team_stats[AwayTeam]['goals_scored'][-5:]),
            'goals_conceded_avg': np.mean(self.team_stats[AwayTeam]['goals_conceded'][-5:]),
            'form': np.mean(self.team_stats[AwayTeam]['last_5_results'][-5:]),
            'points': self.team_stats[AwayTeam]['points']
        }
        
        features = np.array([[ 
            home_stats['goals_scored_avg'],
            home_stats['goals_conceded_avg'],
            away_stats['goals_scored_avg'],
            away_stats['goals_conceded_avg'],
            home_stats['form'],
            away_stats['form'],
            home_stats['points'],
            away_stats['points']
        ]])
        
        # Scale features and predict
        features_scaled = self.scaler.transform(features)
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        return {
            'home_win': probabilities[0],
            'draw': probabilities[1],
            'away_win': probabilities[2]
        }

        
predictor = FootballMatchPredictor('Premier League')
predictor.train("Datasets/season-2425.csv")
result = predictor.predict_match("Arsenal", "Liverpool")
print(result)
