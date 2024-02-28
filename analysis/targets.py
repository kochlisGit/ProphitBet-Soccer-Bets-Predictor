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
        self._target_counts = {}

    def _get_target_counts(self, task: ClassificationTask) -> (pd.Series, list[str]):
        if task not in self._target_counts:
            if task == ClassificationTask.Result:
                self._target_counts[task] = self._input_df['Result'].value_counts()[['H', 'D', 'A']].values
            elif task == ClassificationTask.Over:
                self._target_counts[task] = ((self._input_df['HG'] + self._input_df['AG']) > 2.5).value_counts().values
            else:
                raise NotImplementedError(f'Not implemented target: {task.name}')

        return self._target_counts[task]

    def plot(self, ax, task: ClassificationTask = ClassificationTask.Result, **kwargs):
        target_counts = self._get_target_counts(task=task)
        columns = self._target_names[task]
        sns.barplot(x=columns, y=target_counts, ax=ax)
