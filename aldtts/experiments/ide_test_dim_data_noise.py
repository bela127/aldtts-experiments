from alts.core.experiment_runner import ExperimentRunner
from aldtts.modules.blueprint import IDTBlueprint, indep_exp
from alts.core.oracle.data_source import DataSource


from alts.modules.data_process.process import DataSourceProcess
from alts.modules.oracle.augmentation import NoiseAugmentation

import numpy as np

from alts.modules.data_sampler import KDTreeRegionDataSampler, KDTreeKNNDataSampler
from alts.modules.query.query_sampler import AllResultPoolQuerySampler
from aldtts.modules.multi_sample_test import KWHMultiSampleTest
from aldtts.modules.dependency_test import SampleTest, DependencyMeasureTest, Pearson, Spearmanr, Kendalltau, FIT, XiCor, Hoeffdings, hypoDcorr, hypoHsic, hypoHHG, hypoMGC, hypoKMERF
from aldtts.modules.dependency_measure import dHSIC, dCor
from aldtts.modules.test_interpolation import KNNTestInterpolator
from aldtts.modules.two_sample_test import MWUTwoSampleTest


from alts.modules.oracle.data_source import (
    LineDataSource,
    SquareDataSource,
    PowDataSource,
    ExpDataSource,
    CrossDataSource,
    DoubleLinearDataSource,
    HourglassDataSource,
    ZDataSource,
    ZInvDataSource,
    LinearPeriodicDataSource,
    LinearStepDataSource,
    SineDataSource,
    HypercubeDataSource,
    StarDataSource,
    HyperSphereDataSource,
    GaussianProcessDataSource,
    BrownianProcessDataSource,
    )

dss = [
    LineDataSource,
    SquareDataSource,
    #PowDataSource,
    #ExpDataSource,
    CrossDataSource,
    #DoubleLinearDataSource,
    HourglassDataSource,
    ZDataSource,
    #ZInvDataSource,
    LinearPeriodicDataSource,
    LinearStepDataSource,
    SineDataSource,
    HypercubeDataSource,
    StarDataSource,
    #HyperSphereDataSource,
    #GaussianProcessDataSource,
    #BrownianProcessDataSource,
]

tests = [
    #SampleTest(
    #            query_sampler = AllResultPoolQuerySampler(),
    #            data_sampler = KDTreeRegionDataSampler(0.05),
    #            multi_sample_test=KWHMultiSampleTest()
    #        ),
    #Pearson(),
    #Spearmanr(),
    #Kendalltau(),
    #FIT(),
    #XiCor(),
    #Hoeffdings(),
    #hypoDcorr(),
    #hypoHsic(),
    hypoHHG(),
    hypoMGC(),
    hypoKMERF(),
    DependencyMeasureTest(dependency_measure=dHSIC()),
    DependencyMeasureTest(dependency_measure=dCor())
]

test_interpolators = [
    KNNTestInterpolator(
        test = MWUTwoSampleTest(),
        data_sampler=KDTreeKNNDataSampler(),
    ),
]

def create_blueprints():
    blueprints = []
    ds: DataSource
    for d in range(1, 3):
        for test_interpolator in test_interpolators:
            for test in tests:
                for ds in dss:
                    for noise in np.arange(0, 5, 0.5):
                        blueprint = IDTBlueprint(
                            repeat=1,
                            process=DataSourceProcess(
                                data_source=NoiseAugmentation(data_source=ds(query_shape = (d,), result_shape = (1,)), noise_ratio=noise)
                            ),
                            experiment_modules = indep_exp(
                                                    test=test,
                                                    test_int=test_interpolator
                                                ),
                            exp_name=f"de_{test.__class__.__name__}_{d}_{ds.__name__}_{noise}", # type: ignore
                        )

                        blueprints.append(blueprint)
    return blueprints

blueprints = create_blueprints()

if __name__ == '__main__':
    er = ExperimentRunner(blueprints)
    er.run_experiments()#_parallel()
