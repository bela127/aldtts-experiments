from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing_extensions import Protocol
import matplotlib
from matplotlib import pyplot as plt
from nptyping import NDArray, Shape, Number, Float
import numpy as np
import os
import random


sim_time = 1000
base_path = "/home/bela/Cloud/code/Git/aldtts-experiments/eval"
save_path = "/home/bela/Cloud/code/Git/aldtts-experiments/fig/exp_figures"

@dataclass
class RunRes():
    run_path: str
    run_name: str

    queries:  NDArray[Shape["Batch, N"], Number]
    scores: NDArray[Shape["Batch, 1"], Number]

    actual_queries: NDArray[Shape["Batch, N"], Number]
    result: NDArray[Shape["Batch, 1"], Number]

    pseudo_queries: NDArray[Shape["Batch, S, N"], Number]
    pseudo_scores: NDArray[Shape["Batch, S, 1"], Number]

    pvalue: NDArray[Shape["Batch, 1"], Number]

def calc_measurements(sim_res: RunRes)-> int:
    measurements: int = sim_res.result.shape[0]
    return measurements

def calc_nr_meas(sim_res: RunRes)-> NDArray[Shape["Batch, 1"], Number]:
    measurements = np.arange(1,sim_res.result.shape[0]+1)
    return measurements

def calc_meas_time(sim_res: RunRes) -> NDArray[Shape["Batch, 1"], Number]:
    time = sim_res.query[:,0]
    return time

def calc_mean_std(datas):

    mean = np.mean(datas, axis=-1)
    std = np.std(datas, axis=-1)

    return mean, std

class FileWorker(Protocol):
    def __call__(self, file_path: str, file_name: str) -> Any: ...

def walk_files(path: str, worker: FileWorker):
    for dirpath, dnames, fnames in os.walk(path):
        f: str
        for f in fnames:
            if f.endswith(".npy"):
                yield worker(file_path = dirpath, file_name = f)

class RunWorker(Protocol):
    def __call__(self, run_path: str, run_name: str) -> Any: ...

def walk_runs(path: str, worker: RunWorker):
    for dirpath, dnames, fnames in os.walk(path):
        d: str
        for d in dnames:
            if d.startswith("exp_"):
                yield worker(run_path = dirpath, run_name = d)

class SubExpWorker(Protocol):
    def __call__(self, sub_exp_path: str, sub_exp_name: str) -> Any: ...

def walk_dirs(path: str, worker: SubExpWorker):
    for dirpath, dnames, fnames in os.walk(path):
        f: str
        for d in dnames:
                yield worker(sub_exp_path = dirpath, sub_exp_name = d)

@dataclass
class EvalRes():
    exp_quantity: NDArray[Shape["Batch, 1"], Float]
    mean_p: NDArray[Shape["Batch, 1"], Float]
    std_p: NDArray[Shape["Batch, 1"], Float]

def norm_time(time):
    return time / sim_time

@dataclass
class DataComputer:
    sort_index: int = 3
    skip_points: int = 0
    sub_exp_folder: str = "exp_sub_folder"
    base_path: str = "./eval/exp_folder"

    def load_file(self, file_path, file_name):
        file_data = np.load(os.path.join(file_path, file_name))
        return self.comp_file(file_data)

    def comp_file(self, file_data):
        return file_data

    def load_run(self, run_path, run_name):
        file_path = os.path.join(run_path, run_name)
        file_actual_query = "ActualQuery_query.npy"
        file_actual_score = "ActualQuery_score.npy"
        file_oracle_data = "oracle_data.npy"
        file_pseudo_query = "PseudoQuery_query.npy"
        file_pseudo_score = "PseudoQuery_score.npy"
        file_pvalue = "PValue.npy"
        file_result = "result.npy"

        actual_query = self.load_file(file_path, file_actual_query)
        actual_score = self.load_file(file_path, file_actual_score)
        oracle_data = self.load_file(file_path, file_oracle_data)
        pseudo_query = self.load_file(file_path, file_pseudo_query)
        pseudo_score = self.load_file(file_path, file_pseudo_score)
        pvalue = self.load_file(file_path, file_pvalue)
        result = self.load_file(file_path, file_result)

        run_res = RunRes(
            run_path=run_path,
            run_name=run_name,

            queries = actual_query,
            scores = actual_score,

            actual_queries = oracle_data,
            result = result[:,-1:],

            pseudo_queries = pseudo_query,
            pseudo_scores = pseudo_score,

            pvalue = pvalue,
        )
        return self.comp_run(run_res)


    def comp_run(self, run_res: RunRes):
        p = run_res.pvalue
        return p


    def load_sub_exp(self, sub_exp_path: str, sub_exp_name: str):
        if sub_exp_name.startswith(self.sub_exp_folder):
            exp_quantity_str = sub_exp_name.removeprefix(self.sub_exp_folder)
            if exp_quantity_str:
                exp_quantity = float(exp_quantity_str)
            else:
                exp_quantity = None

            run_data = [run_result for run_result in walk_runs(path=os.path.join(sub_exp_path, sub_exp_name), worker=self.load_run)]

            data = exp_quantity, run_data

            return self.comp_sub_exp(data)

    def comp_sub_exp(self, data):
        exp_quantity, ps = data

        ps = np.asarray(ps).T

        mean_p, std_p = calc_mean_std(ps)

        quant_p_std = (exp_quantity, mean_p, std_p)
        
        return quant_p_std

    def load_exp(self):
        data = [i_r_v for i_r_v in walk_dirs(path=self.base_path, worker=self.load_sub_exp) if i_r_v is not None]
        return self.comp_exp(data)

    def comp_exp(self, loaded_data):
        exp_quantity, p_mean, p_std = zip(*loaded_data)

        eval_res = EvalRes(exp_quantity=np.asarray(exp_quantity), mean_p=np.asarray(p_mean), std_p=np.asarray(p_std))
        return eval_res
    
#--------- Plot config --------


linestyle = {
     'dashdot': 'dashdot',  # Same as '-.'
     'solid':                 (0, ()),
     #'loosely dotted':        (0, (1, 10)),
     'dotted':                (0, (1, 1)),
     #'long dash with offset': (5, (10, 3)),
     #'loosely dashed':        (0, (5, 10)),
     'dashed':                (0, (5, 5)),
     'densely dashed':        (0, (5, 1)),

     #'loosely dashdotted':    (0, (3, 10, 1, 10)),
     'dashdotted':            (0, (3, 5, 1, 5)),
     'densely dashdotted':    (0, (3, 1, 1, 1)),

     'dashdotdotted':         (0, (3, 5, 1, 5, 1, 5)),
     #'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
     'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1)),
}

colors = [
            "#88CCEE",
            "#CC6677",
            "#DDCC77",
            "#117733",
            "#332288",
            "#AA4499",
            "#44AA99",
            "#999933",
            "#882255",
            "#661100",
            "#888888"
        ]

marker = [
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "s",
    "*",
    "+",
    "x",
    "d",
]

appr_style = {
    "MS": (colors[5], linestyle['dashed'], marker[5]),
    "CI": (colors[7], linestyle['solid'], marker[0]),
    "CEm": (colors[2], linestyle['dotted'], marker[1]),
    "CEo": (colors[6], linestyle['dashdotted'], marker[2]),
    "CEw": (colors[3], linestyle['densely dashdotdotted'], marker[4]),
    "CAL": (colors[1], linestyle['dashdot'], marker[13]),
    "CALm": (colors[8], linestyle['densely dashdotted'], marker[0]),
    "BR": (colors[0], linestyle['dashdotdotted'], marker[11]),
    "IB": (colors[4], linestyle['densely dashed'], marker[10]),
    "BRt": (colors[9], linestyle['dashdotdotted'],marker[12]),
}


def save(fig, name, path):
    #plt.title(name)
    fig.tight_layout()
    loc = os.path.join(path,f"{name}.svg")
    fig.savefig(loc, format="svg", bbox_inches='tight', transparent="True", pad_inches=0)
    fig.clf()

def set_size(width: float|str = "paper_2c", fraction:float=1, subplots=(1, 1), hfrac=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'paper_2c':
        width_pt = 252
    elif width == 'paper':
        width_pt = 516
    elif width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        if isinstance(width, float):
            width_pt = width
        else:
            raise ValueError(f"A with of {width} is a unrecognized with.")

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio: float = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * 1.2 * hfrac * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

def create_fig(width="paper_2c", fraction:float =1, subplots=(1, 1), hfrac:float=1):
    plt.style.use('seaborn-v0_8-paper')

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "text.latex.preamble": r'\usepackage{amssymb}',
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }

    plt.rcParams.update(tex_fonts)


    fig, axs = plt.subplots(subplots[0], subplots[1], figsize=set_size(width=width, fraction=fraction, subplots=subplots, hfrac=hfrac))
    #axs.set_prop_cycle(line_cycler)
    return fig, axs




#----------- Plotting ---------

def plot_rmse_over_exp_quant(ax, eval_res: EvalRes, exp_quant_name = "exp_quant", color=None):
    
    ax.plot(eval_res.exp_quantity, eval_res.mean_rmse, label=f"Mean {exp_quant_name}", color=color)
    ax.fill_between(
        eval_res.exp_quantity,
        eval_res.mean_rmse - eval_res.std_rmse*1.96,
        eval_res.mean_rmse + eval_res.std_rmse*1.96,
        alpha=0.1,
        color=color,
        #label=r"$\pm$ 1 std. dev.",
    )
    ax.set_xlabel(exp_quant_name)
    ax.set_ylabel("$RMSE$")

def plot_rmse_over_stdsel(ax, eval_res: EvalRes, exp_quant_name = "exp_quant", color=None):
    
    ax.plot(eval_res.exp_quantity, eval_res.mean_rmse, label=f"{exp_quant_name}", color=color)
    ax.fill_between(
        eval_res.exp_quantity,
        eval_res.mean_rmse - eval_res.std_rmse*1.96,
        eval_res.mean_rmse + eval_res.std_rmse*1.96,
        alpha=0.1,
        color=color,
        #label=r"$\pm$ 1 std. dev.",
    )
    ax.set_xlabel(r"$\sqrt{v_{target}}$")
    ax.set_ylabel("$RMSE$")


def plot_p_over_mean_meas(ax, eval_res: EvalRes, exp_quant_name = "exp_quant", print_var =False, color=None, style=None, marker=None):
    xs = np.arange(0, eval_res.mean_p.shape[1])
    if marker is not None:
        ax.plot(xs, eval_res.mean_p[0], color=color, linestyle=style)

        x,y = random.choice(list(zip(xs, eval_res.mean_p[0])))
        ax.scatter(x, y, marker=marker, label=exp_quant_name, color=color)
    else:
        ax.plot(xs, eval_res.mean_p[0], label=exp_quant_name, color=color, linestyle=style)

    if print_var:
        ax.fill_between(
            xs,
            eval_res.mean_p[0] - eval_res.std_p[0]*1.96,
            eval_res.mean_p[0] + eval_res.std_p[0]*1.96,
            alpha=0.2,
            color=color,
            label=r"$\pm$ 1 std. dev.",
            linewidth=0,
        )
    else:
        ax.fill_between(
            xs,
            eval_res.mean_p[0] - eval_res.std_p[0]*1.96,
            eval_res.mean_p[0] + eval_res.std_p[0]*1.96,
            alpha=0.2,
            color=color,
            linewidth=0,
        )

    ax.set_xlabel("Sample size")
    ax.set_ylabel('$P-Value$')

def plot_sum_meas_over_time(ax, meas_res, exp_quant_name = "exp_quant", color=None):

    for exp_quantity, times, mean_measures, std_meas in zip(*meas_res):
    
        ax.plot(times, mean_measures, label=f"Mean for {exp_quant_name}={exp_quantity:.3f}", color=color)
        ax.fill_between(
            times,
            mean_measures - std_meas*1.96,
            mean_measures + std_meas*1.96,
            alpha=0.1,
            color=color,
            #label=r"$\pm$ 1 std. dev.",
        )
        ax.set_xlabel("time")
        ax.set_ylabel("total acquired measurements")

def plot_meas_over_time(ax, meas_res, exp_quant_name = "exp_quant", color=None):
    colors = []

    for i, (exp_quantity, times, mean_measures, std_meas) in enumerate(zip(*meas_res)):
        if isinstance(color, list):
            color = color[i]
    
        ax.plot(times, mean_measures/times, label=f"{exp_quant_name}={exp_quantity:.3f}", color=color)
        colors.append(plt.gca().lines[-1].get_color())
        ax.fill_between(
            times,
            (mean_measures - std_meas*1.96)/times,
            (mean_measures + std_meas*1.96)/times,
            alpha=0.1,
            color=plt.gca().lines[-1].get_color(),
            #label=r"$\pm$ 1 std. dev.",
        )
    ax.set_xlabel("time $t$")
    ax.set_ylabel("$m_{su}$ = meas. / $su$")
    return colors


def plot_meas_per_step_vs_exp_quant(ax, meas_res, exp_quant_name = "exp_quant", color=None):
    exp_quantities, _,_,_ = meas_res

    mean_meass = []
    mean_stds = []
    for exp_quantity, times, mean_measures, std_meas in zip(*meas_res):
        mean_meas = np.median(mean_measures/times)
        mean_std = np.median(std_meas/times)
        mean_meass.append(mean_meas)
        mean_stds.append(mean_std)
        if mean_meas ==  1.0:
            print(mean_meas)

    #for exp_quantity, mean_meas in list(zip(exp_quantities, mean_meass))[::2]:
    #    print(exp_quantity, mean_meas)

    exp_quantities = np.asarray(exp_quantities)
    mean_meass = np.asarray(mean_meass)
    mean_stds = np.asarray(mean_stds)
    
    ax.plot(exp_quantities, mean_meass, label=f"{exp_quant_name}", color=color)
    ax.fill_between(
        exp_quantities,
        mean_meass - mean_stds*1.96,
        mean_meass + mean_stds*1.96,
        alpha=0.1,
        color=color,
        #label=r"$\pm$ 1 std. dev.",
    )
    ax.set_xlabel(r"$v_{target}$")
    ax.set_ylabel(r"$\bar{m}_{su} = \mathbb{E}[$meas.$ / su]$")

def plot_pred_vs_gt(sim_res: RunRes):
    plt.plot(sim_res.time, sim_res.estimation)
    plt.plot(sim_res.time, sim_res.gt)