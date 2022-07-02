from database.entities.leagues import League
import csv


class LeagueReader:
    @staticmethod
    def read_all_leagues(leagues_filepath: str) -> list:
        with open(leagues_filepath, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            return [
                League(
                    country=row[0],
                    league_name=row[1],
                    base_url=row[2],
                    year_start=int(row[3]),
                    league_type=row[4],
                    fixtures_url=row[5]
                ) for row in reader
            ]
