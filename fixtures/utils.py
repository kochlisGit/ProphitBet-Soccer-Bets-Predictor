from fuzzywuzzy import process


def match_fixture_teams(parsed_home_teams: list, parsed_away_teams: list, unique_league_teams: set) -> (list, list):
    matches_teams = []
    num_teams = len(parsed_home_teams)

    for parsed_team in parsed_home_teams + parsed_away_teams:
        try:
            extracted_team = process.extractOne(parsed_team, unique_league_teams)[0]
            matches_teams.append(extracted_team)
            unique_league_teams.remove(extracted_team)
        except Exception as e:
            print(e)
            print('LOL')
    return matches_teams[: num_teams], matches_teams[num_teams:]
