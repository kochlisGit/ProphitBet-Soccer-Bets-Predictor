from fuzzywuzzy import process


def match_fixture_teams(
    fix_home_teams: list, fix_away_teams: list, unique_league_teams: set
) -> (list, list):
    available_league_teams = unique_league_teams.copy()
    team_similarity_dict = {}

    for fix_team in fix_home_teams + fix_away_teams:
        if fix_team in team_similarity_dict.keys():
            continue
        extracted_league_team = process.extractOne(fix_team, available_league_teams)[0]
        team_similarity_dict[fix_team] = extracted_league_team
        available_league_teams.remove(extracted_league_team)

    fix_home_teams = [team_similarity_dict[team] for team in fix_home_teams]
    fix_away_teams = [team_similarity_dict[team] for team in fix_away_teams]
    return fix_home_teams, fix_away_teams
