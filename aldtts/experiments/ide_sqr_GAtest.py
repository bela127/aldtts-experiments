from alts.core.experiment_runner import ExperimentRunner
from aldtts.modules.blueprint import IDTBlueprint
from alts.modules.data_process.process import DataSourceProcess
from alts.modules.oracle.data_source import SquareDataSource
from alts.modules.oracle.augmentation import NoiseAugmentation
from aldtts.modules.experiment_modules import InterventionDependencyExperiment
from alts.modules.data_sampler import KDTreeKNNDataSampler, KDTreeRegionDataSampler
from alts.modules.query.query_sampler import AllResultPoolQuerySampler, LatinHypercubeQuerySampler
from aldtts.modules.multi_sample_test import KWHMultiSampleTest
from aldtts.modules.dependency_test import SampleTest
from alts.core.query.query_selector import ResultQuerySelector
from alts.modules.query.query_optimizer import GAQueryOptimizer
from aldtts.modules.selection_criteria import PValueSelectionCriteria
from aldtts.modules.test_interpolation import KNNTestInterpolator
from aldtts.modules.two_sample_test import MWUTwoSampleTest
from aldtts.modules.query.query_decider import UnpackAllQueryDecider


blueprint = IDTBlueprint(
    process = DataSourceProcess(data_source=NoiseAugmentation(data_source=SquareDataSource((1,),(1,)), noise_ratio=3.0)),
    experiment_modules=InterventionDependencyExperiment(
                query_selector=ResultQuerySelector(
                    query_optimizer=GAQueryOptimizer(
                        selection_criteria=PValueSelectionCriteria(),
                    ),
                    query_decider=UnpackAllQueryDecider(),
                    ),
                dependency_test=SampleTest(
                    query_sampler = AllResultPoolQuerySampler(),
                    data_sampler = KDTreeRegionDataSampler(0.05),
                    multi_sample_test=KWHMultiSampleTest()
                ),
                test_interpolator = KNNTestInterpolator(
                    test = MWUTwoSampleTest(),
                    data_sampler=KDTreeKNNDataSampler(),
                ),
            ),
    exp_name="ide_sqr",
)
blueprints = [blueprint]

if __name__ == '__main__':
    er = ExperimentRunner(blueprints)
    er.run_experiments()
