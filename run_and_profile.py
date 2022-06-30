from aldtts.experiments.ide_lin import blueprint
from alts.core.experiment_runner import ExperimentRunner

if __name__ == '__main__': 
    er = ExperimentRunner([blueprint])
    er.run_experiments()