"""Bookmaker identities; no synthetic prices are attributed to real bookmakers."""
from backend.app.services.betting_integrity import valid_odds

BOOKMAKER_CONFIGS = {
    "HOLLYWOODBETS": {"name": "Hollywoodbets", "base_url": "https://www.hollywoodbets.net", "brand_color": "#7c3aed"},
    "BETWAY": {"name": "Betway South Africa", "base_url": "https://sports.betway.co.za", "brand_color": "#10b981"},
}


class SASportsbookScraper:
    @staticmethod
    def calculate_margin(odds):
        if not odds or not all(valid_odds(value) for value in odds):
            return None
        return round((sum(1 / value for value in odds) - 1) * 100, 2)

    def generate_calibrated_sa_odds_for_bookmaker(self, **kwargs):
        raise ValueError("Model-derived prices are not observed bookmaker odds")

    async def scrape_all_sportsbooks_for_match(self, **kwargs):
        return {}  # No verified odds feed is connected; never fabricate prices.


sa_scraper = SASportsbookScraper()
