from alts.core.experiment_runner import ExperimentRunner
from aldtts.modules.blueprint import DTBlueprint
from aldtts.modules.evaluator import PlotTestPEvaluator, PlotQueriesEvaluator
from alts.modules.data_process.process import DataSourceProcess
from alts.modules.oracle.data_source import SquareDataSource
from alts.modules.oracle.augmentation import NoiseAugmentation


blueprint = DTBlueprint(
    process = DataSourceProcess(data_source=NoiseAugmentation(data_source=SquareDataSource((1,),(1,)), noise_ratio=3.0)),
    exp_name="de_sqr",
)
blueprints = [blueprint]

if __name__ == '__main__':
    er = ExperimentRunner(blueprints)
    er.run_experiments()
