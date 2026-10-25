import dtreeviz
import numpy as np
import pandas as pd
from typing import Any, Dict
from sklearn.tree import DecisionTreeClassifier
from src.core.analysis.analyzers.analyzer import Analyzer
from src.core.preprocessing.utils.targets import TargetType


class RuleExtractionAnalyzer(Analyzer):
    """ Analyzer that utilizes decision trees to form classification rules. """

    def __init__(self):
        super().__init__(name='Rules')

    def analyze(self, df: pd.DataFrame, metadata: pd.DataFrame) -> Dict[str, Any]:
        # Removing samples (rows) with nan values.
        valid_ids = df.notna().all(axis=1)
        valid_df = df[valid_ids]
        valid_metadata = metadata[valid_ids]
        columns = df.columns

        # Constructing inputs.
        x = valid_df.to_numpy(dtype=np.float32)

        rules_dict = {}
        for target in TargetType:
            mapper = target.get_mapper()
            y = valid_metadata[target.value].map(mapper).to_numpy(dtype=np.int32)
            tree = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=5, random_state=0)
            tree.fit(x, y)
            try:
                tree_model = dtreeviz.model(
                    model=tree,
                    X_train=x,
                    y_train=y,
                    feature_names=columns,
                    target_name=target.value,
                    class_names=list(mapper.keys())
                )
            except Exception as _:
                rules_dict = None
                break
            rules_dict[target.value] = tree_model
        return {'rules_dict': rules_dict}
