from alts.modules.queried_data_pool import FlatQueriedDataPool
from alts.modules.data_sampler import KDTreeKNNDataSampler, KDTreeRegionDataSampler
from alts.core.oracle.oracle import Oracle
from alts.core.query.query_optimizer import NoQueryOptimizer
from alts.modules.query.query_sampler import LastQuerySampler, RandomChoiceQuerySampler, UniformQuerySampler, LatinHypercubeQuerySampler
from aldtts.modules.selection_criteria import QueryTestNoSelectionCritera
from aldtts.modules.test_interpolation import KNNTestInterpolator
from aldtts.modules.two_sample_test import MWUTwoSampleTest
from aldtts.modules.experiment_modules import DependencyExperiment
from alts.modules.oracle.augmentation import NoiseAugmentation
from alts.modules.stopping_criteria import LearningStepStoppingCriteria
from alts.core.blueprint import Blueprint
from alts.modules.oracle.data_source import LineDataSource, SquareDataSource
from alts.modules.evaluator import LogNewDataPointsEvaluator, PlotNewDataPointsEvaluator, PrintNewDataPointsEvaluator, PlotQueryDistEvaluator
from aldtts.modules.evaluator import PlotScoresEvaluator, PlotQueriesEvaluator, PlotTestPEvaluator, BoxPlotTestPEvaluator, LogPValueEvaluator
from aldtts.modules.dependency_test import DependencyTest
from aldtts.modules.multi_sample_test import KWHMultiSampleTest


blueprint = Blueprint(
    repeat=100,
    stopping_criteria= LearningStepStoppingCriteria(290),
    oracle = Oracle(
        data_source=LineDataSource((2,),(1,)),
        augmentation= NoiseAugmentation(noise_ratio=2.0)
    ),
    queried_data_pool=FlatQueriedDataPool(),
    initial_query_sampler=LatinHypercubeQuerySampler(num_queries=10),
    query_optimizer=NoQueryOptimizer(
        selection_criteria=QueryTestNoSelectionCritera(),
        num_queries=4,
        query_sampler=UniformQuerySampler(),
    ),
    experiment_modules=DependencyExperiment(
        dependency_test=DependencyTest(
            query_sampler = LastQuerySampler(),
            data_sampler = KDTreeRegionDataSampler(0.05),
            multi_sample_test=KWHMultiSampleTest()
            ),
        ),
    #evaluators=[PlotQueryDistEvaluator(), PlotNewDataPointsEvaluator(), PlotQueriesEvaluator(), PlotTestPEvaluator(), BoxPlotTestPEvaluator()],
    evaluators=[LogPValueEvaluator(folder="log_de")],
    #evaluators=[PlotNewDataPointsEvaluator(), PlotScoresEvaluator(), PlotQueriesEvaluator(), PlotTestPEvaluator(), BoxPlotTestPEvaluator()],

)