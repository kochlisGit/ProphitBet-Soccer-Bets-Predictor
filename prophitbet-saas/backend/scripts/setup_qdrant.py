"""
Qdrant Collections Initialization Script for ProphitBet.
Sets up vector collections for team profiles, match fixtures, historical matches, and predictions.
"""

import json
import urllib.request
import urllib.error

QDRANT_URL = "http://localhost:6333"

COLLECTIONS = [
    {
        "name": "team_profiles",
        "description": "Team embeddings for soccer analytics, style of play, and attack/defense ratings",
        "vector_size": 128,
        "distance": "Cosine",
    },
    {
        "name": "match_fixtures",
        "description": "Upcoming and active match fixture situational vectors for similarity matching",
        "vector_size": 128,
        "distance": "Cosine",
    },
    {
        "name": "historical_matches",
        "description": "Historical match feature vectors used for nearest-neighbor similarity predictions",
        "vector_size": 128,
        "distance": "Cosine",
    },
    {
        "name": "predictions_store",
        "description": "Prediction embeddings, outcome probabilities, and betting expected-value clusters",
        "vector_size": 128,
        "distance": "Cosine",
    },
]


def make_request(url: str, method: str = "GET", data: dict = None):
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8") if data else None,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            return json.loads(body)
        except Exception:
            return {"error": str(e), "body": body}


def setup_collections():
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    health = make_request(f"{QDRANT_URL}/collections")
    print(f"Current Qdrant collections: {health.get('result', {}).get('collections', [])}")

    for col in COLLECTIONS:
        name = col["name"]
        size = col["vector_size"]
        distance = col["distance"]

        # Check if collection exists
        info = make_request(f"{QDRANT_URL}/collections/{name}")
        if info.get("status") == "ok":
            print(f"[EXISTS] Collection '{name}' already exists.")
            continue

        # Create collection
        payload = {
            "vectors": {
                "size": size,
                "distance": distance,
            }
        }
        res = make_request(f"{QDRANT_URL}/collections/{name}", method="PUT", data=payload)
        print(f"[CREATED] Collection '{name}' (vectors={size}, distance={distance}): {res.get('status')}")

        # Create payload indexes for fast filtering
        index_fields = ["league_id", "league_name", "team_name", "match_date"]
        for field in index_fields:
            idx_payload = {
                "field_name": field,
                "field_schema": "keyword",
            }
            make_request(f"{QDRANT_URL}/collections/{name}/index", method="PUT", data=idx_payload)

    # Populate initial sample team profiles so the dashboard has rich data
    sample_teams = [
        {"name": "Arsenal", "league": "Premier League", "country": "England", "rating": 88},
        {"name": "Manchester City", "league": "Premier League", "country": "England", "rating": 92},
        {"name": "Liverpool", "league": "Premier League", "country": "England", "rating": 90},
        {"name": "Real Madrid", "league": "La Liga", "country": "Spain", "rating": 93},
        {"name": "FC Barcelona", "league": "La Liga", "country": "Spain", "rating": 89},
        {"name": "Bayern Munich", "league": "Bundesliga", "country": "Germany", "rating": 91},
        {"name": "Paris Saint-Germain", "league": "Ligue 1", "country": "France", "rating": 89},
        {"name": "Inter Milan", "league": "Serie A", "country": "Italy", "rating": 88},
    ]

    points = []
    for i, t in enumerate(sample_teams, 1):
        # Deterministic normalized 128-dim vector
        import math
        vec = [(math.sin(i * 0.5 + j * 0.1) + 1.0) / 2.0 for j in range(128)]
        # Normalize
        norm = math.sqrt(sum(x * x for x in vec))
        vec = [x / norm for x in vec]

        points.append({
            "id": i,
            "vector": vec,
            "payload": {
                "team_name": t["name"],
                "league_name": t["league"],
                "country": t["country"],
                "elo_rating": t["rating"],
                "updated_at": "2026-09-08",
            }
        })

    upsert_res = make_request(
        f"{QDRANT_URL}/collections/team_profiles/points",
        method="PUT",
        data={"points": points}
    )
    print(f"[UPSERT] Inserted {len(points)} team profile vectors: {upsert_res.get('status')}")

    # Final verification
    final = make_request(f"{QDRANT_URL}/collections")
    col_names = [c["name"] for c in final.get("result", {}).get("collections", [])]
    print(f"\n[OK] Qdrant is fully initialized with collections: {col_names}")


if __name__ == "__main__":
    setup_collections()
