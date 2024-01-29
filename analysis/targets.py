import pandas as pd
import seaborn as sns
from analysis.analyzer import FeatureAnalyzer
from models.tasks import ClassificationTask


class TargetAnalyzer(FeatureAnalyzer):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df=df)

        self._target_names = {
            ClassificationTask.Result: ['Home (1)', 'Draw (x)', 'Away (2)'],
            ClassificationTask.Over: ['Under (2.5)', 'Over (2.5)']
        }
        self._targets = {}

    def _get_targets(self, task: ClassificationTask) -> (pd.Series, list[str]):
        if task not in self._targets:
            if task == ClassificationTask.Result:
                self._targets[task] = self._input_df['Result'].replace({'H': 0, 'D': 1, 'A': 2})
            elif task == ClassificationTask.Over:
                self._targets[task] = ((self._input_df['HG'] + self._input_df['AG']) > 2.5).astype(int)
            else:
                raise NotImplementedError(f'Not implemented target: {task.name}')

        return self._targets[task], self._target_names[task]

    def plot(self, ax, task: ClassificationTask = ClassificationTask.Result, **kwargs):
        targets, columns = self._get_targets(task=task)
        sns.barplot(x=columns, y=targets.value_counts(), ax=ax)
