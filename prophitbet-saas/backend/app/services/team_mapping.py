"""
Team name normalization and mapping utilities.

Provides canonical team name mappings to bridge differences between:
- FootyStats fixture team names
- football-data.co.uk historical data team names
- Other data sources

Uses football-data.co.uk team names as the canonical reference since they're used
for training data.
"""

from typing import Optional

# Canonical team name mappings
# Key: FootyStats/scraped team name -> Value: football-data.co.uk canonical team name
TEAM_NAME_MAPPINGS = {
    # England - Premier League
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Newcastle": "Newcastle United",
    "Newcastle Utd": "Newcastle United",
    "Tottenham": "Tottenham Hotspur",
    "Spurs": "Tottenham Hotspur",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
    "Nott'm Forest": "Nottingham Forest",
    "Nottingham Forest": "Nottingham Forest",
    "Brighton": "Brighton and Hove Albion",
    "Brighton & Hove Albion": "Brighton and Hove Albion",
    "Sheffield United": "Sheffield United",
    "Sheffield Utd": "Sheffield United",
    "Luton": "Luton Town",
    "Luton Town": "Luton Town",
    "Burnley": "Burnley",
    "Fulham": "Fulham",
    "Brentford": "Brentford",
    "Crystal Palace": "Crystal Palace",
    "Aston Villa": "Aston Villa",
    "Villa": "Aston Villa",
    
    # England - Championship
    "Leicester": "Leicester City",
    "Leeds": "Leeds United",
    "Leeds United": "Leeds United",
    "Southampton": "Southampton",
    "Sunderland": "Sunderland",
    "Middlesbrough": "Middlesbrough",
    "Boro": "Middlesbrough",
    "Coventry": "Coventry City",
    "Coventry City": "Coventry City",
    "Watford": "Watford",
    "Norwich": "Norwich City",
    "Norwich City": "Norwich City",
    "West Brom": "West Bromwich Albion",
    "West Bromwich Albion": "West Bromwich Albion",
    "WBA": "West Bromwich Albion",
    "Hull": "Hull City",
    "Hull City": "Hull City",
    "Cardiff": "Cardiff City",
    "Cardiff City": "Cardiff City",
    "Swansea": "Swansea City",
    "Swansea City": "Swansea City",
    "Bristol City": "Bristol City",
    "QPR": "Queens Park Rangers",
    "Queens Park Rangers": "Queens Park Rangers",
    "Preston": "Preston North End",
    "Preston North End": "Preston North End",
    "Blackburn": "Blackburn Rovers",
    "Blackburn Rovers": "Blackburn Rovers",
    "Stoke": "Stoke City",
    "Stoke City": "Stoke City",
    "Birmingham": "Birmingham City",
    "Birmingham City": "Birmingham City",
    "Millwall": "Millwall",
    "Rotherham": "Rotherham United",
    "Rotherham United": "Rotherham United",
    "Huddersfield": "Huddersfield Town",
    "Huddersfield Town": "Huddersfield Town",
    "Sheffield Wed": "Sheffield Wednesday",
    "Sheffield Wednesday": "Sheffield Wednesday",
    "Plymouth": "Plymouth Argyle",
    "Plymouth Argyle": "Plymouth Argyle",
    "Ipswich": "Ipswich Town",
    "Ipswich Town": "Ipswich Town",
    
    # England - League One
    "Portsmouth": "Portsmouth",
    "Derby": "Derby County",
    "Derby County": "Derby County",
    "Barnsley": "Barnsley",
    "Bolton": "Bolton Wanderers",
    "Bolton Wanderers": "Bolton Wanderers",
    "Peterborough": "Peterborough United",
    "Peterborough United": "Peterborough United",
    "Oxford": "Oxford United",
    "Oxford United": "Oxford United",
    "Wycombe": "Wycombe Wanderers",
    "Wycombe Wanderers": "Wycombe Wanderers",
    "Charlton": "Charlton Athletic",
    "Charlton Athletic": "Charlton Athletic",
    "Lincoln": "Lincoln City",
    "Lincoln City": "Lincoln City",
    "Shrewsbury": "Shrewsbury Town",
    "Shrewsbury Town": "Shrewsbury Town",
    "Fleetwood": "Fleetwood Town",
    "Fleetwood Town": "Fleetwood Town",
    "Exeter": "Exeter City",
    "Exeter City": "Exeter City",
    "Cheltenham": "Cheltenham Town",
    "Cheltenham Town": "Cheltenham Town",
    "Burton": "Burton Albion",
    "Burton Albion": "Burton Albion",
    "Cambridge": "Cambridge United",
    "Cambridge United": "Cambridge United",
    "MK Dons": "Milton Keynes Dons",
    "Milton Keynes Dons": "Milton Keynes Dons",
    "Morecambe": "Morecambe",
    "Accrington": "Accrington Stanley",
    "Accrington Stanley": "Accrington Stanley",
    
    # England - League Two
    "Stockport": "Stockport County",
    "Stockport County": "Stockport County",
    "Wrexham": "Wrexham",
    "Mansfield": "Mansfield Town",
    "Mansfield Town": "Mansfield Town",
    "Bromley": "Bromley",
    "Sutton": "Sutton United",
    "Sutton United": "Sutton United",
    "Grimsby": "Grimsby Town",
    "Grimsby Town": "Grimsby Town",
    "Hartlepool": "Hartlepool United",
    "Hartlepool United": "Hartlepool United",
    "Rochdale": "Rochdale",
    "Notts County": "Notts County",
    "Doncaster": "Doncaster Rovers",
    "Doncaster Rovers": "Doncaster Rovers",
    "Crewe": "Crewe Alexandra",
    "Crewe Alexandra": "Crewe Alexandra",
    "Swindon": "Swindon Town",
    "Swindon Town": "Swindon Town",
    "Salford": "Salford City",
    "Salford City": "Salford City",
    "Bradford": "Bradford City",
    "Bradford City": "Bradford City",
    "Tranmere": "Tranmere Rovers",
    "Tranmere Rovers": "Tranmere Rovers",
    "Newport": "Newport County",
    "Newport County": "Newport County",
    "Harrogate": "Harrogate Town",
    "Harrogate Town": "Harrogate Town",
    "Crawley": "Crawley Town",
    "Crawley Town": "Crawley Town",
    "Walsall": "Walsall",
    "Colchester": "Colchester United",
    "Colchester United": "Colchester United",
    "Gillingham": "Gillingham",
    "AFC Wimbledon": "AFC Wimbledon",
    "Wimbledon": "AFC Wimbledon",
    
    # Spain - La Liga
    "Real Madrid": "Real Madrid",
    "Barcelona": "Barcelona",
    "Atletico Madrid": "Atletico Madrid",
    "Atlético Madrid": "Atletico Madrid",
    "Sevilla": "Sevilla",
    "Valencia": "Valencia",
    "Villarreal": "Villarreal",
    "Athletic Bilbao": "Athletic Bilbao",
    "Athletic": "Athletic Bilbao",
    "Real Sociedad": "Real Sociedad",
    "Real Betis": "Real Betis",
    "Betis": "Real Betis",
    "Osasuna": "Osasuna",
    "Getafe": "Getafe",
    "Celta Vigo": "Celta Vigo",
    "Celta de Vigo": "Celta Vigo",
    "Rayo Vallecano": "Rayo Vallecano",
    "Rayo": "Rayo Vallecano",
    "Mallorca": "Mallorca",
    "Alaves": "Alaves",
    "Alavés": "Alaves",
    "Girona": "Girona",
    "Cadiz": "Cadiz",
    "Cádiz": "Cadiz",
    "Las Palmas": "Las Palmas",
    "Granada": "Granada",
    "Almeria": "Almeria",
    "Almería": "Almeria",
    
    # Italy - Serie A
    "Juventus": "Juventus",
    "Inter": "Inter Milan",
    "Inter Milan": "Inter Milan",
    "AC Milan": "AC Milan",
    "Milan": "AC Milan",
    "Napoli": "Napoli",
    "Roma": "Roma",
    "AS Roma": "Roma",
    "Lazio": "Lazio",
    "Atalanta": "Atalanta",
    "Fiorentina": "Fiorentina",
    "Bologna": "Bologna",
    "Torino": "Torino",
    "Monza": "Monza",
    "Genoa": "Genoa",
    "Sassuolo": "Sassuolo",
    "Udinese": "Udinese",
    "Cagliari": "Cagliari",
    "Empoli": "Empoli",
    "Verona": "Hellas Verona",
    "Hellas Verona": "Hellas Verona",
    "Lecce": "Lecce",
    "Salernitana": "Salernitana",
    "Frosinone": "Frosinone",
    
    # Germany - Bundesliga
    "Bayern Munich": "Bayern Munich",
    "Bayern München": "Bayern Munich",
    "Borussia Dortmund": "Borussia Dortmund",
    "Dortmund": "Borussia Dortmund",
    "BVB": "Borussia Dortmund",
    "RB Leipzig": "RB Leipzig",
    "Leipzig": "RB Leipzig",
    "Bayer Leverkusen": "Bayer Leverkusen",
    "Leverkusen": "Bayer Leverkusen",
    "Borussia M'gladbach": "Borussia Monchengladbach",
    "Borussia Monchengladbach": "Borussia Monchengladbach",
    "Borussia Mönchengladbach": "Borussia Monchengladbach",
    "Gladbach": "Borussia Monchengladbach",
    "VfB Stuttgart": "VfB Stuttgart",
    "Stuttgart": "VfB Stuttgart",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "Frankfurt": "Eintracht Frankfurt",
    "Wolfsburg": "Wolfsburg",
    "VfL Wolfsburg": "Wolfsburg",
    "Union Berlin": "Union Berlin",
    "Freiburg": "Freiburg",
    "SC Freiburg": "Freiburg",
    "Mainz": "Mainz 05",
    "Mainz 05": "Mainz 05",
    "Hoffenheim": "Hoffenheim",
    "TSG Hoffenheim": "Hoffenheim",
    "Augsburg": "Augsburg",
    "FC Augsburg": "Augsburg",
    "Werder Bremen": "Werder Bremen",
    "Bremen": "Werder Bremen",
    "Bochum": "Bochum",
    "VfL Bochum": "Bochum",
    "Heidenheim": "Heidenheim",
    "FC Heidenheim": "Heidenheim",
    "Darmstadt": "Darmstadt",
    "SV Darmstadt": "Darmstadt",
    
    # France - Ligue 1
    "PSG": "Paris Saint-Germain",
    "Paris Saint-Germain": "Paris Saint-Germain",
    "Paris SG": "Paris Saint-Germain",
    "Marseille": "Marseille",
    "Olympique Marseille": "Marseille",
    "Lyon": "Lyon",
    "Olympique Lyon": "Lyon",
    "Monaco": "Monaco",
    "AS Monaco": "Monaco",
    "Lille": "Lille",
    "LOSC Lille": "Lille",
    "Rennes": "Rennes",
    "Stade Rennes": "Rennes",
    "Nice": "Nice",
    "OGC Nice": "Nice",
    "Lens": "Lens",
    "RC Lens": "Lens",
    "Reims": "Reims",
    "Stade Reims": "Reims",
    "Montpellier": "Montpellier",
    "Toulouse": "Toulouse",
    "Brest": "Brest",
    "Stade Brest": "Brest",
    "Strasbourg": "Strasbourg",
    "Nantes": "Nantes",
    "FC Nantes": "Nantes",
    "Lorient": "Lorient",
    "FC Lorient": "Lorient",
    "Clermont": "Clermont",
    "Clermont Foot": "Clermont",
    "Le Havre": "Le Havre",
    "Metz": "Metz",
    "FC Metz": "Metz",
    "Auxerre": "Auxerre",
    "AJ Auxerre": "Auxerre",
    
    # Portugal - Liga 1
    "Benfica": "Benfica",
    "SL Benfica": "Benfica",
    "Porto": "Porto",
    "FC Porto": "Porto",
    "Sporting CP": "Sporting CP",
    "Sporting Lisbon": "Sporting CP",
    "Sporting": "Sporting CP",
    "Braga": "Braga",
    "SC Braga": "Braga",
    "Guimaraes": "Guimaraes",
    "Vitoria Guimaraes": "Guimaraes",
    "Vitória Guimarães": "Guimaraes",
    "Arouca": "Arouca",
    "FC Arouca": "Arouca",
    "Famalicao": "Famalicao",
    "FC Famalicao": "Famalicao",
    "Casa Pia": "Casa Pia",
    "Casa Pia AC": "Casa Pia",
    "Chaves": "Chaves",
    "GD Chaves": "Chaves",
    "Estoril": "Estoril",
    "Estoril Praia": "Estoril",
    "Gil Vicente": "Gil Vicente",
    "Moreirense": "Moreirense",
    "Portimonense": "Portimonense",
    "Rio Ave": "Rio Ave",
    "Rio Ave FC": "Rio Ave",
    "Vizela": "Vizela",
    "FC Vizela": "Vizela",
    
    # Netherlands - Eredivisie
    "Ajax": "Ajax",
    "PSV": "PSV Eindhoven",
    "PSV Eindhoven": "PSV Eindhoven",
    "Feyenoord": "Feyenoord",
    "AZ": "AZ Alkmaar",
    "AZ Alkmaar": "AZ Alkmaar",
    "Twente": "Twente",
    "FC Twente": "Twente",
    "Utrecht": "Utrecht",
    "FC Utrecht": "Utrecht",
    "Heerenveen": "Heerenveen",
    "SC Heerenveen": "Heerenveen",
    "Vitesse": "Vitesse",
    "Vitesse Arnhem": "Vitesse",
    "Zwolle": "Zwolle",
    "PEC Zwolle": "Zwolle",
    "NEC": "NEC Nijmegen",
    "NEC Nijmegen": "NEC Nijmegen",
    "Sparta Rotterdam": "Sparta Rotterdam",
    "Sparta": "Sparta Rotterdam",
    "Groningen": "Groningen",
    "FC Groningen": "Groningen",
    "Go Ahead Eagles": "Go Ahead Eagles",
    "GA Eagles": "Go Ahead Eagles",
    "Heracles": "Heracles Almelo",
    "Heracles Almelo": "Heracles Almelo",
    "Almere City": "Almere City",
    "RKC Waalwijk": "RKC Waalwijk",
    "RKC": "RKC Waalwijk",
    "Volendam": "Volendam",
    "FC Volendam": "Volendam",
    "Excelsior": "Excelsior",
    "SBV Excelsior": "Excelsior",
    
    # Belgium - Jupiler League
    "Club Brugge": "Club Brugge",
    "Club Brugge KV": "Club Brugge",
    "Union Saint-Gilloise": "Union Saint-Gilloise",
    "Union SG": "Union Saint-Gilloise",
    "Antwerp": "Antwerp",
    "Royal Antwerp": "Antwerp",
    "Genk": "Genk",
    "KRC Genk": "Genk",
    "Gent": "Gent",
    "KAA Gent": "Gent",
    "Anderlecht": "Anderlecht",
    "RSC Anderlecht": "Anderlecht",
    "Cercle Brugge": "Cercle Brugge",
    "Standard Liege": "Standard Liege",
    "Standard Liège": "Standard Liege",
    "Charleroi": "Charleroi",
    "Sporting Charleroi": "Charleroi",
    "KV Mechelen": "Mechelen",
    "Mechelen": "Mechelen",
    "KV Kortrijk": "Kortrijk",
    "Kortrijk": "Kortrijk",
    "OH Leuven": "OH Leuven",
    "Oud-Heverlee Leuven": "OH Leuven",
    "Westerlo": "Westerlo",
    "KVC Westerlo": "Westerlo",
    "Eupen": "Eupen",
    "KAS Eupen": "Eupen",
    "Sint-Truiden": "Sint-Truiden",
    "STVV": "Sint-Truiden",
    
    # Other leagues - common abbreviations
    "FC": "",
    "SC": "",
    "AC": "",
    "AS": "",
    "Real": "Real",
    "CF": "",
    "CD": "",
    "SD": "",
    "UD": "",
    "RCD": "",
    "CA": "",
    "C": "",
    "AFC": "",
    "B": "",
}

# Reverse mapping for lookup by canonical name
_CANONICAL_TO_ALIASES = {}
for alias, canonical in TEAM_NAME_MAPPINGS.items():
    if canonical not in _CANONICAL_TO_ALIASES:
        _CANONICAL_TO_ALIASES[canonical] = []
    _CANONICAL_TO_ALIASES[canonical].append(alias)


def normalize_team_name(raw_name: str, canonical_names: list[str]) -> Optional[str]:
    """
    Normalize a raw/scraped team name to a canonical name from the training data.
    
    Args:
        raw_name: The team name as scraped from fixtures (e.g., "Man City")
        canonical_names: List of team names that exist in the training data
        
    Returns:
        The canonical team name if found, None otherwise
    """
    if not raw_name:
        return None
    
    raw = raw_name.strip()
    if not raw:
        return None
    
    # Direct match in canonical names
    if raw in canonical_names:
        return raw
    
    # Check explicit mapping
    if raw in TEAM_NAME_MAPPINGS:
        mapped = TEAM_NAME_MAPPINGS[raw]
        if mapped in canonical_names:
            return mapped
    
    # Case-insensitive match
    raw_lower = raw.lower()
    for canonical in canonical_names:
        if canonical.lower() == raw_lower:
            return canonical
    
    # Check aliases for each canonical name
    for canonical in canonical_names:
        if canonical in _CANONICAL_TO_ALIASES:
            for alias in _CANONICAL_TO_ALIASES[canonical]:
                if alias.lower() == raw_lower:
                    return canonical
    
    # Fuzzy matching using difflib
    import difflib
    close = difflib.get_close_matches(raw, canonical_names, n=1, cutoff=0.75)
    if close:
        return close[0]
    
    # Partial match (raw contained in canonical or vice versa)
    for canonical in canonical_names:
        canonical_lower = canonical.lower()
        if raw_lower in canonical_lower or canonical_lower in raw_lower:
            return canonical
    
    return None


def get_all_mappings() -> dict[str, str]:
    """Get all team name mappings for inspection/debugging."""
    return TEAM_NAME_MAPPINGS.copy()


def add_mapping(alias: str, canonical: str) -> None:
    """Add a new team name mapping at runtime."""
    TEAM_NAME_MAPPINGS[alias] = canonical
    if canonical not in _CANONICAL_TO_ALIASES:
        _CANONICAL_TO_ALIASES[canonical] = []
    if alias not in _CANONICAL_TO_ALIASES[canonical]:
        _CANONICAL_TO_ALIASES[canonical].append(alias)