import os
import numpy as np
from utils import plot_p_over_mean_meas, save, DataComputer, base_path, save_path, create_fig, appr_style
from matplotlib import pyplot as plt

fig, ax = create_fig(subplots=(1,1), width="paper_2c", fraction=1)

datasource = "LineDataSource"
noise = "2.0"

folders = [
    f"de_Pearson_1_{datasource}_{noise}",
    f"de_SampleTest_1_{datasource}_{noise}",
    f"de_Spearmanr_1_{datasource}_{noise}",
    f"de_Kendalltau_1_{datasource}_{noise}",
    f"de_XiCor_1_{datasource}_{noise}",
    f"de_Hoeffdings_1_{datasource}_{noise}",
    f"de_hypoDcorr_1_{datasource}_{noise}",
    f"de_hypoHsic_1_{datasource}_{noise}",
    f"de_hypoHHG_1_{datasource}_{noise}",
    #f"de_hypoMGC_1_{datasource}_{noise}",
    #f"de_hypoKMERF_1_{datasource}_{noise}",
    #f"de_DependencyMeasureTest_1_{datasource}_{noise}",
    #f"de_DependencyMeasureTest_1_{datasource}_{noise}",
]

for foler in folders:

    dc =DataComputer(
        base_path=os.path.join(base_path,"./de_test_dim_data_noise"),
        sub_exp_folder=foler,
    )
    base_eval_res = dc.load_exp()

    plot_p_over_mean_meas(ax, base_eval_res, exp_quant_name=foler)

ax.set_ylim(0, 1)
ax.set_title("P value over sample size")
plt.legend()
save(fig= fig, name=f"p_value over query_nr for {datasource}",path=save_path)
