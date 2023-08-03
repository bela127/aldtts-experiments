from dataclasses import dataclass
from utils import DataComputer, walk_runs
import os

@dataclass
class RunComputer(DataComputer):
    test: bool = True

    def load_run(self, run_path, run_name):
        file_path = os.path.join(run_path, run_name)
        old_file_path = os.path.join(file_path, f"PseudoQuery_query_xx.npy")
        new_file_path = os.path.join(file_path, "PseudoQuery_score.npy")

        if not self.test:
            try:
                os.rename(old_file_path, new_file_path)
            except FileNotFoundError:
                print(old_file_path, "not found, probably already renamed.")

        message = f"renamed '{old_file_path}' to '{new_file_path}'!"
        return run_path, message


    def comp_sub_exp(self, data):
        return data[1]
    
    def comp_exp(self, loaded_data):
        return loaded_data


dc = RunComputer(
    base_path="/home/bela/Cloud/code/Git/aldtts-experiments/eval/de_lin",
    sub_exp_folder="de_lin",
    test=False

)
eval_res = dc.load_exp()
print(eval_res)
