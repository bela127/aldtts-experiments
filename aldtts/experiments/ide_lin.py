from alts.core.experiment_runner import ExperimentRunner
from aldtts.modules.blueprint import IDTBlueprint



blueprint = IDTBlueprint(
    exp_name="ide_lin",
)
blueprints = [blueprint]

if __name__ == '__main__':
    er = ExperimentRunner(blueprints)
    er.run_experiments()