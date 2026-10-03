"""Thin market naming and journal adapter for shared ML targets."""
from backend.app.config import setup_ml_path

setup_ml_path()
from src.preprocessing.utils.target import TargetType, parse_target, target_labels


def market_name(market):
    target = parse_target(market)
    if target == TargetType.RESULT:
        return 'Match result (1X2)'
    if target == TargetType.BTTS:
        return 'Both teams to score'
    if target == TargetType.OVER_UNDER:
        return 'Total goals 2.5'
    prefix, line = target.value.rsplit('-', 1)
    return f"{dict(goals='Total goals', corners='Total corners', **{'home-goals': 'Home goals', 'away-goals': 'Away goals', 'shots-target': 'Total shots on target'})[prefix]} {line}"


def market_selection(market, label):
    target = parse_target(market)
    if label not in target_labels(target):
        raise ValueError('Outcome does not belong to the prediction market')
    if target == TargetType.RESULT:
        return label
    if target == TargetType.BTTS:
        return f'BTTS {label}'
    if target == TargetType.OVER_UNDER:
        return f'{label} 2.5'
    prefix, line = target.value.rsplit('-', 1)
    kind = {'goals': '', 'corners': 'Corners ', 'home-goals': 'Home goals ',
            'away-goals': 'Away goals ', 'shots-target': 'Shots on target '}[prefix]
    return f'{kind}{label} {line}'


SELECTION_TARGETS = {market_selection(target, label): (target.value, label)
                     for target in TargetType for label in target_labels(target)}


def settle_prediction_selection(selection, prediction):
    """Only verified same-market results may settle statistics-based bets."""
    from backend.app.services.betting_integrity import settlement
    market = parse_target(getattr(prediction, 'market_type', None) or 'result').value
    if market == "result":
        return settlement(selection, prediction.actual_result, prediction.actual_score)
    selected = SELECTION_TARGETS.get(selection)
    if selected and selected[0] == market:
        if prediction.actual_result not in target_labels(market):
            return None
        return 'WON' if selected[1] == prediction.actual_result else 'LOST'
    # A corners prediction's goal score is not a verified corners result for
    # another line. Require the matching market or leave the slip pending.
    return None
