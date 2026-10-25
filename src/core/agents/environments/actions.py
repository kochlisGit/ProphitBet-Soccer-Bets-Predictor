from enum import Enum
import numpy as np


class Action(Enum):
    """ The supported betting actions.
        Given a match, the agent can choose on of the following actions: SKIP | SOLO | ACCUMULATOR.
        If SKIP, the agent skips the match entirely. If SOLO option, it places a separate bet on the specified match.
        If ACCUMULATOR option, the payouts of multiple bets are multiplied. It usually returns large payouts, but
        the entire bet is lost if a single bet is lost.

        Supported Types:
            0. SKIP
            1. SOLO-1
            2. SOLO-X
            3. SOLO-2
            4. SOLO-1X
            5. SOLO-12
            6. SOLO-X2
            7. MULTIPLE-1
            8. MULTIPLE-X
            9. MULTIPLE-2
            10. MULTIPLE-1X
            11. MULTIPLE-12
            12. MULTIPLE-X2
    """

    SKIP = 0
    SOLO_1 = 1
    SOLO_X = 2
    SOLO_2 = 3
    SOLO_1X = 4
    SOLO_12 = 5
    SOLO_X2 = 6
    ACCUMULATOR_1 = 7
    ACCUMULATOR_X = 8
    ACCUMULATOR_2 = 9
    ACCUMULATOR_1X = 10
    ACCUMULATOR_12 = 11
    ACCUMULATOR_X2 = 12

    @staticmethod
    def get_name(action: int) -> str:
        """ Returns the action name.

            Args
            ----
            :param action: (int) action value.
        """

        if action == 0:
            return 'SKIP'
        elif action == 1:
            return '1 (Solo)'
        elif action == 2:
            return 'X (Solo)'
        elif action == 3:
            return '2 (Solo)'
        elif action == 4:
            return '1X (Solo)'
        elif action == 5:
            return '12 (Solo)'
        elif action == 6:
            return 'X2 (Solo)'
        elif action == 7:
            return '1 (Multiple)'
        elif action == 8:
            return 'X (Multiple)'
        elif action == 9:
            return '2 (Multiple)'
        elif action == 10:
            return '1X (Multiple)'
        elif action == 11:
            return '12 (Multiple)'
        elif action == 12:
            return 'X2 (Multiple)'
        else:
            raise NotImplementedError(f'Invalid action: "{action}"')

    @classmethod
    def to_one_hot_vector(cls, action: int) -> np.ndarray:
        """ Converts the given action into a binary vector.

            :param action: (int) action to be converted into a binary vector.
        """

        one_hot_vector = np.zeros(shape=(len(cls),), dtype=np.float32)
        one_hot_vector[action] = 1.0
        return one_hot_vector
