from alts.core.experiment_runner import ExperimentRunner
from aldtts.modules.blueprint import IDTBlueprint
from aldtts.modules.evaluator import PlotTestPEvaluator, PlotQueriesEvaluator
from alts.modules.data_process.process import DataSourceProcess
from alts.modules.oracle.data_source import IndependentDataSource
from alts.modules.oracle.augmentation import NoiseAugmentation


blueprint = IDTBlueprint(
    process = DataSourceProcess(data_source=NoiseAugmentation(data_source=IndependentDataSource((1,),(1,)), noise_ratio=3.0)),
    exp_name="ide_ind",
)
blueprints = [blueprint]

if __name__ == '__main__':
    er = ExperimentRunner(blueprints)
    er.run_experiments()
