"""Linear API Integration Service for ProphitBet.

Provides automated tracking for predictions, match scores, +EV bets,
system health, and developer tasks directly inside Linear.
"""

import json
import logging
import os
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LINEAR_API_KEY = os.getenv("LINEAR_API_KEY", "")
LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"


class LinearService:
    @classmethod
    def _query(cls, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a GraphQL query/mutation against Linear API."""
        if not LINEAR_API_KEY:
            logger.warning("LINEAR_API_KEY not configured. Skipping Linear API call.")
            return {}

        payload = json.dumps({"query": query, "variables": variables or {}}).encode("utf-8")
        req = urllib.request.Request(
            LINEAR_GRAPHQL_URL,
            data=payload,
            headers={
                "Authorization": LINEAR_API_KEY,
                "Content-Type": "application/json",
                "User-Agent": "ProphitBet-LinearService/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                if "errors" in result:
                    logger.error(f"Linear GraphQL errors: {result['errors']}")
                return result
        except Exception as e:
            logger.error(f"Failed to execute Linear query: {e}")
            return {}

    @classmethod
    def get_team_id(cls) -> Optional[str]:
        """Fetch primary team ID."""
        q = """
        query {
          teams {
            nodes {
              id
              name
              key
            }
          }
        }
        """
        res = cls._query(q)
        teams = res.get("data", {}).get("teams", {}).get("nodes", [])
        if teams:
            return teams[0]["id"]
        return None

    @classmethod
    def get_or_create_project(cls, project_name: str = "ProphitBet Soccer Predictor") -> Optional[str]:
        """Get or create the ProphitBet Linear Project."""
        team_id = cls.get_team_id()
        if not team_id:
            return None

        # Check existing projects
        q_list = """
        query {
          projects {
            nodes {
              id
              name
            }
          }
        }
        """
        res = cls._query(q_list)
        for proj in res.get("data", {}).get("projects", {}).get("nodes", []):
            if proj["name"].lower() == project_name.lower():
                return proj["id"]

        # Create project if it doesn't exist
        mutation = """
        mutation CreateProject($name: String!, $teamIds: [String!]!) {
          projectCreate(input: {
            name: $name,
            teamIds: $teamIds,
            description: "Autonomous ProphitBet Predictions, Match Scores and System Operations"
          }) {
            success
            project {
              id
              name
            }
          }
        }
        """
        res_create = cls._query(mutation, {"name": project_name, "teamIds": [team_id]})
        proj_data = res_create.get("data", {}).get("projectCreate", {}).get("project")
        if proj_data:
            logger.info(f"Created Linear project '{project_name}' (ID: {proj_data['id']})")
            return proj_data["id"]
        return None

    @classmethod
    def create_prediction_issue(cls, alert: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a Linear Issue for a high-value +EV prediction or match."""
        team_id = cls.get_team_id()
        if not team_id:
            return None

        project_id = cls.get_or_create_project()

        title = f"⚽ [{alert.get('league', 'Soccer')}] {alert.get('match', 'Match')} — Pick: {alert.get('pick_name', 'H')} (+EV {alert.get('ev_edge_pct', 0)}%)"
        description = (
            f"## 🚨 ProphitBet Automated Prediction Alert\n\n"
            f"- **Match**: {alert.get('match')}\n"
            f"- **League**: {alert.get('league')}\n"
            f"- **Kickoff**: {alert.get('match_date')}\n\n"
            f"### Quantitative Model Metrics\n"
            f"- **Recommended Pick**: `{alert.get('pick_name')} ({alert.get('pick')})`\n"
            f"- **Market Odds**: `@{alert.get('market_odds', 0.0):.2f}`\n"
            f"- **Win Probability**: `{alert.get('win_probability_pct', 0)}%`\n"
            f"- **Expected Value Edge (+EV)**: `+{alert.get('ev_edge_pct', 0)}%`\n"
            f"- **Quarter-Kelly Bankroll Stake**: `{alert.get('quarter_kelly_stake_pct', 0)}%`\n"
            f"- **MiroFish Consensus**: `{alert.get('consensus_level', 'N/A')}`\n\n"
            f"*Status: Pending Match Completion*"
        )

        mutation = """
        mutation CreateIssue($teamId: String!, $projectId: String, $title: String!, $description: String!) {
          issueCreate(input: {
            teamId: $teamId,
            projectId: $projectId,
            title: $title,
            description: $description,
            priority: 2
          }) {
            success
            issue {
              id
              identifier
              title
              url
            }
          }
        }
        """
        params = {
            "teamId": team_id,
            "projectId": project_id,
            "title": title,
            "description": description,
        }
        res = cls._query(mutation, params)
        issue_data = res.get("data", {}).get("issueCreate", {}).get("issue")
        if issue_data:
            logger.info(f"Created Linear Issue {issue_data['identifier']}: {issue_data['url']}")
        return issue_data

    @classmethod
    def post_score_settlement_comment(
        cls, issue_id: str, home_score: int, away_score: int, result: str, won: bool, ev_gained: float
    ) -> bool:
        """Post final score settlement comment and update status on a Linear issue."""
        outcome_symbol = "✅ WIN" if won else "❌ LOSS"
        comment_body = (
            f"### 🏁 Match Completed & Score Settled\n\n"
            f"- **Final Score**: {home_score} - {away_score}\n"
            f"- **Match Outcome**: `{result}`\n"
            f"- **Prediction Result**: **{outcome_symbol}**\n"
            f"- **EV Realized**: `{ev_gained:+.2f}%`\n"
        )

        mutation = """
        mutation CreateComment($issueId: String!, $body: String!) {
          commentCreate(input: {
            issueId: $issueId,
            body: $body
          }) {
            success
            comment {
              id
            }
          }
        }
        """
        res = cls._query(mutation, {"issueId": issue_id, "body": comment_body})
        return res.get("data", {}).get("commentCreate", {}).get("success", False)

    @classmethod
    def create_dev_task(cls, title: str, description: str, priority: int = 2) -> Optional[Dict[str, Any]]:
        """Create a development task in Linear for Antigravity or team members."""
        team_id = cls.get_team_id()
        if not team_id:
            return None

        project_id = cls.get_or_create_project()

        mutation = """
        mutation CreateIssue($teamId: String!, $projectId: String, $title: String!, $description: String!, $priority: Int!) {
          issueCreate(input: {
            teamId: $teamId,
            projectId: $projectId,
            title: $title,
            description: $description,
            priority: $priority
          }) {
            success
            issue {
              id
              identifier
              title
              url
            }
          }
        }
        """
        res = cls._query(mutation, {
            "teamId": team_id,
            "projectId": project_id,
            "title": title,
            "description": description,
            "priority": priority,
        })
        return res.get("data", {}).get("issueCreate", {}).get("issue")
