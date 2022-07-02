from py_stringmatching.similarity_measure.monge_elkan import MongeElkan
from py_stringmatching.similarity_measure.token_sort import TokenSort
import numpy as np


class TeamSimilarityMatching:
    def __init__(self):
        self._monge_elkan = MongeElkan()
        self._token_sort = TokenSort()

    def _get_matching_team_name_ind(self, team_name: str, all_teams: list) -> str:
        monge_elkan_sims = np.float64(
            [self._monge_elkan.get_raw_score(team_name.split(' '), team.split(' '))*100 for team in all_teams]
        )
        token_sort_sims = np.float64(
            [self._token_sort.get_raw_score(team_name, team)*0.5 for team in all_teams]
        )
        similarity_scores = (monge_elkan_sims + token_sort_sims)/2
        return all_teams[np.argmax(similarity_scores)]

    def match_teams(self, fixture_matches: list, all_teams: list) -> list:
        matches = []

        for home_team, away_team in fixture_matches:
            matching_home_team = self._get_matching_team_name_ind(team_name=home_team, all_teams=all_teams)
            matching_away_team = self._get_matching_team_name_ind(team_name=away_team, all_teams=all_teams)
            matches.append((matching_home_team, matching_away_team))
        return matches
