import pandas as pd
import numpy as np
from typing import Dict


def prepare_features(matches_df: pd.DataFrame, team_stats: Dict) -> pd.DataFrame:
    """
    Calculate match features and update team_stats in-place.
    """
    # initialize stats if empty
    if not team_stats:
        teams = set(matches_df['HomeTeam']) | set(matches_df['AwayTeam'])
        for t in teams:
            team_stats[t] = {'goals_scored':[], 'goals_conceded':[], 'last_5_results':[], 'points':0}

    features = []
    matches_df = matches_df.sort_values('Date')

    for _, m in matches_df.iterrows():
        H, A = m['HomeTeam'], m['AwayTeam']
        hg, ag = m['FTHG'], m['FTAG']
        # compute averages
        def avg(lst): return np.mean(lst[-5:]) if lst else 0
        home = {
            'goals_scored_avg': avg(team_stats[H]['goals_scored']),
            'goals_conceded_avg': avg(team_stats[H]['goals_conceded']),
            'form': avg(team_stats[H]['last_5_results']),
            'points': team_stats[H]['points']
        }
        away = {
            'goals_scored_avg': avg(team_stats[A]['goals_scored']),
            'goals_conceded_avg': avg(team_stats[A]['goals_conceded']),
            'form': avg(team_stats[A]['last_5_results']),
            'points': team_stats[A]['points']
        }
        features.append({
            'HomeTeam_goals_scored_avg': home['goals_scored_avg'],
            'HomeTeam_goals_conceded_avg': home['goals_conceded_avg'],
            'AwayTeam_goals_scored_avg': away['goals_scored_avg'],
            'AwayTeam_goals_conceded_avg': away['goals_conceded_avg'],
            'HomeTeam_form': home['form'], 'AwayTeam_form': away['form'],
            'HomeTeam_points': home['points'], 'AwayTeam_points': away['points'],
            'result': get_result(hg, ag)
        })
        update_team_stats(team_stats, H, A, hg, ag)

    return pd.DataFrame(features)


def update_team_stats(team_stats: Dict, H: str, A: str, hg: int, ag: int) -> None:
    """
    Append the latest match result to stats.
    """
    team_stats[H]['goals_scored'].append(hg)
    team_stats[H]['goals_conceded'].append(ag)
    team_stats[A]['goals_scored'].append(ag)
    team_stats[A]['goals_conceded'].append(hg)

    if hg > ag:
        team_stats[H]['last_5_results'].append(1)
        team_stats[A]['last_5_results'].append(0)
        team_stats[H]['points'] += 3
    elif hg < ag:
        team_stats[H]['last_5_results'].append(0)
        team_stats[A]['last_5_results'].append(1)
        team_stats[A]['points'] += 3
    else:
        team_stats[H]['last_5_results'].append(0.5)
        team_stats[A]['last_5_results'].append(0.5)
        team_stats[H]['points'] += 1
        team_stats[A]['points'] += 1


def get_result(hg: int, ag: int) -> int:
    """Map goals to 0: home win, 1: draw, 2: away win."""
    if hg > ag: return 0
    if hg < ag: return 2
    return 1