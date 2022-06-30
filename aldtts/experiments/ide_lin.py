from alts.modules.queried_data_pool import FlatQueriedDataPool
from alts.modules.data_sampler import KDTreeKNNDataSampler, KDTreeRegionDataSampler
from alts.core.oracle.oracle import Oracle
from alts.modules.query.query_optimizer import MaxMCQueryOptimizer, ProbWeightedMCQueryOptimizer
from alts.modules.query.query_sampler import LastQuerySampler, UniformQuerySampler, LatinHypercubeQuerySampler
from aldtts.modules.selection_criteria import PValueSelectionCriteria, PValueUncertaintySelectionCriteria, TestScoreUncertaintySelectionCriteria, TestScoreSelectionCriteria, PValueDensitySelectionCriteria
from aldtts.modules.test_interpolation import KNNTestInterpolator
from aldtts.modules.two_sample_test import MWUTwoSampleTest
from aldtts.modules.experiment_modules import InterventionDependencyExperiment
from alts.modules.oracle.augmentation import NoiseAugmentation
from alts.modules.stopping_criteria import LearningStepStoppingCriteria
from alts.core.blueprint import Blueprint
from alts.modules.oracle.data_source import LineDataSource, SquareDataSource
from alts.modules.evaluator import LogNewDataPointsEvaluator, PlotNewDataPointsEvaluator, PrintNewDataPointsEvaluator, PlotQueryDistEvaluator
from aldtts.modules.evaluator import LogActualQueryScoresEvaluator, LogPValueEvaluator, LogScoresEvaluator, PlotScoresEvaluator, PlotQueriesEvaluator, PlotTestPEvaluator, BoxPlotTestPEvaluator
from aldtts.modules.dependency_test import DependencyTest
from aldtts.modules.multi_sample_test import KWHMultiSampleTest


blueprint = Blueprint(
    repeat=1,
    stopping_criteria= LearningStepStoppingCriteria(290),
    oracle = Oracle(
        data_source=LineDataSource((1,),(1,)),
        augmentation= NoiseAugmentation(noise_ratio=2.0)
    ),
    queried_data_pool=FlatQueriedDataPool(),
    initial_query_sampler=LatinHypercubeQuerySampler(num_queries=10),
    query_optimizer=MaxMCQueryOptimizer(
        selection_criteria=PValueSelectionCriteria(),
        num_queries=4,
        query_sampler=LatinHypercubeQuerySampler(),
        num_tries=2000
    ),
    experiment_modules=InterventionDependencyExperiment(
        test_interpolator=KNNTestInterpolator(
            test=MWUTwoSampleTest(),
            data_sampler=KDTreeKNNDataSampler(50,sample_size_data_fraction=10)
            ),
        dependency_test=DependencyTest(
            query_sampler = LastQuerySampler(),
            data_sampler = KDTreeRegionDataSampler(0.05),
            multi_sample_test=KWHMultiSampleTest()
            ),
        ),
    #evaluators=[PlotQueryDistEvaluator(), PlotNewDataPointsEvaluator(), PlotScoresEvaluator(), PlotQueriesEvaluator(), PlotTestPEvaluator(), BoxPlotTestPEvaluator()],
    evaluators=[LogPValueEvaluator(folder="log_ide"),LogScoresEvaluator(folder="log_gt_scores"),LogActualQueryScoresEvaluator(folder="log_scores_over_time")],
    #evaluators=[ PlotQueriesEvaluator(), PlotScoresEvaluator()],

)