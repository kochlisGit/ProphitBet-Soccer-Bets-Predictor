from src.gui.windows.models.trainers.decisiontree import DecisionTreeTrainerDialog
from src.gui.windows.models.trainers.discriminant import DiscriminantTrainerDialog
from src.gui.windows.models.trainers.extremeboosting import ExtremeBoostingTrainerDialog
from src.gui.windows.models.trainers.knn import KNNTrainerDialog
from src.gui.windows.models.trainers.logistic import LogisticRegressionTrainerDialog
from src.gui.windows.models.trainers.naivebayes import NaiveBayesTrainerDialog
from src.gui.windows.models.trainers.randomforest import RandomForestTrainerDialog

try:
    from src.gui.windows.models.trainers.nn import NeuralNetworkTrainerDialog
except ImportError:
    NeuralNetworkTrainerDialog = None
from src.gui.windows.models.trainers.svm import SVMTrainerDialog
