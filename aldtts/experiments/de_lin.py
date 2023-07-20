from alts.core.experiment_runner import ExperimentRunner
from aldtts.modules.blueprint import DTBlueprint


blueprint = DTBlueprint()
blueprints = [blueprint]

if __name__ == '__main__':
    er = ExperimentRunner(blueprints)
    er.run_experiments()
