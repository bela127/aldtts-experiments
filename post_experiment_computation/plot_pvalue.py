import os
import numpy as np
from utils import plot_p_over_mean_meas, save, DataComputer, base_path, save_path, create_fig, appr_style
from matplotlib import pyplot as plt

fig, ax = create_fig(subplots=(1,1), width="paper_2c", fraction=1)

dc =DataComputer(
    base_path=os.path.join(base_path,"./de_lin"),
    sub_exp_folder="de_lin",
)
base_eval_res = dc.load_exp()

plot_p_over_mean_meas(ax, base_eval_res, exp_quant_name=r"MulSam_lin", color=appr_style["MS"][0], marker=appr_style["MS"][2])

dc =DataComputer(
    base_path=os.path.join(base_path,"./de_sqr"),
    sub_exp_folder="de_sqr",
)
base_eval_res = dc.load_exp()

plot_p_over_mean_meas(ax, base_eval_res, exp_quant_name=r"MulSam_sqr", color=appr_style["CEm"][0], marker=appr_style["CEm"][2])

dc =DataComputer(
    base_path=os.path.join(base_path,"./de_data_dim_test_noise"),
    sub_exp_folder="de_RandomUniformDataSource_1_SampleTest_0.0",
)
base_eval_res = dc.load_exp()

plot_p_over_mean_meas(ax, base_eval_res, exp_quant_name=r"MulSam_uni", color=appr_style["CEo"][0], marker=appr_style["CEo"][2])

ax.set_ylim(0, 1)
ax.set_title("P value over sample size")
plt.legend()
save(fig= fig, name="p_value over query_nr",path=save_path)
