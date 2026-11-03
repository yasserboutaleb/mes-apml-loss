#!/usr/bin/env python3
"""Regenerate five publication-style figures from analytic or controlled toy data.

IMPORTANT: All outputs are SYNTHETIC / ANALYTIC, not real dataset results.
No input files, networks, secret data, checkpoints, or background jobs are used.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np
import torch

from mes_apml import apml_loss, mes_loss

NAVY = "#14253D"
TEAL = "#0C8A84"
ORANGE = "#D88A27"
MUTED = "#65758A"
LIGHT = "#E6EDF1"
LILAC = "#7B70BD"
COLORS = ["#3174B6", "#0D927D", "#E0A031", "#D45D66", "#8B71BB", "#74839B", "#497B73", "#A95B89"]


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "axes.edgecolor": "#8796A6",
        "axes.labelcolor": NAVY,
        "text.color": NAVY,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 120,
    })


def save(fig, out: Path, basename: str):
    # The code and analytic arrays, rather than the SVG hash IDs, are the source.
    fig.savefig(out / (basename + ".svg"), bbox_inches="tight", pad_inches=0.24)
    fig.savefig(out / (basename + ".png"), dpi=220, bbox_inches="tight", pad_inches=0.24)
    plt.close(fig)


def toy_fixed_embedding_experiment():
    """Optimize *only* proxy weights for two objectives, with fixed embeddings.

    This is a controlled example of different losses on the same synthetic data,
    NOT an end-to-end comparison of trained encoders or a performance study.
    """
    seed = 321
    n_per_class = 60
    epochs = 275
    lr = 0.035
    beta = 0.008
    rng = np.random.default_rng(seed)
    anchors = np.array([
        [1.90, 0.20, 0.65],
        [-0.55, 1.80, -0.25],
        [-1.85, -0.90, 0.60],
        [0.55, -1.45, -1.10],
    ], dtype=np.float64)
    labels = np.repeat(np.arange(4), n_per_class)
    embeddings = anchors[labels] + rng.normal(0, 0.25, size=(len(labels), 3))
    initial = anchors * 0.49 + rng.normal(0, .19, size=(4,3))
    z_t = torch.tensor(embeddings, dtype=torch.float64)
    y_t = torch.tensor(labels, dtype=torch.int64)
    torch.set_num_threads(1)
    outputs = {}
    for name, strength in (("APML", 0.0), ("APML + MES", beta)):
        proxies = torch.nn.Parameter(torch.tensor(initial.copy(), dtype=torch.float64))
        optimizer = torch.optim.Adam([proxies], lr=lr)
        # Same optimizer config, batch, seed, initial values, and step count.
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            objective = apml_loss(z_t, y_t, proxies) + strength * mes_loss(proxies)
            objective.backward()
            optimizer.step()
        with torch.no_grad():
            weights_np = proxies.detach().cpu().numpy().copy()
            outputs[name] = {
                "weights": weights_np,
                "apml": float(apml_loss(z_t, y_t, proxies).item()),
                "mes": float(mes_loss(proxies).item()),
            }
    return {
        "seed": seed,
        "n_per_class": n_per_class,
        "steps": epochs,
        "learning_rate": lr,
        "beta": beta,
        "anchors": anchors,
        "embeddings": embeddings,
        "labels": labels,
        "initial": initial,
        "results": outputs,
    }


def equal_3d(ax, mid, half):
    ax.set(xlim=(mid[0]-half,mid[0]+half),
           ylim=(mid[1]-half,mid[1]+half),
           zlim=(mid[2]-half,mid[2]+half))
    ax.set_box_aspect((1,1,1))
    ax.view_init(elev=20, azim=-54)
    ax.set_xlabel("axis 1", labelpad=2)
    ax.set_ylabel("axis 2", labelpad=2)
    ax.set_zlabel("axis 3", labelpad=2)
    ax.tick_params(pad=0, labelsize=7)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("#F5F8FA")
        axis.line.set_color(LIGHT)


def figure_01(out, t):
    xyz = np.vstack([t["embeddings"], t["initial"]] + [v["weights"] for v in t["results"].values()])
    mid = (xyz.max(axis=0) + xyz.min(axis=0)) / 2
    half = float((xyz.max(axis=0)-xyz.min(axis=0)).max())*.61
    fig = plt.figure(figsize=(12.5, 6.6))
    for i,(name, rec) in enumerate(t["results"].items(), 1):
        ax = fig.add_subplot(1,2,i,projection="3d")
        for cid in range(4):
            mask = t["labels"] == cid
            ax.scatter(*t["embeddings"][mask].T, color=COLORS[cid], s=5, alpha=.22, depthshade=False)
            ax.scatter(*rec["weights"][cid], color=COLORS[cid], marker="D", s=90,
                       edgecolors=NAVY, linewidth=.8, depthshade=False)
            ax.text(*(rec["weights"][cid]+0.12), f"P{cid+1}", fontsize=8, color=NAVY)
        equal_3d(ax,mid,half)
        ax.set_title(name + (f"  (β = {t['beta']})" if i==2 else "  (β = 0)"),
                     pad=7, weight="bold", color=NAVY)
        fig.text(.25 if i == 1 else .75, .135,
                 f"APML = {rec['apml']:.3f}    |    MES = {rec['mes']:.3f}",
                 ha="center", fontsize=10.5, color=MUTED)
    fig.subplots_adjust(bottom=.22, top=.87, wspace=.08)
    fig.suptitle("Same fixed toy embeddings · different proxy regularization", fontsize=17,
                 color=NAVY, weight="bold", y=.975)
    fig.text(.5,.045,"SYNTHETIC / COMPUTED  •  Embeddings fixed, only proxies optimized  •  Matched 3D axes  •  Not a benchmark",
             fontsize=9.2, color=MUTED, ha="center")
    save(fig, out, "01_toy_comparison")


def figure_02(out):
    fig, ax = plt.subplots(figsize=(14,6.5))
    ax.set(xlim=(0,14),ylim=(0,6.5))
    ax.axis("off")

    def box(x,y,w,h,title,sub="",face="#F7FAFB",edge=LIGHT):
        patch=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.10,rounding_size=0.13",
                             facecolor=face,edgecolor=edge,linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x+w/2,y+h*.64,title,ha="center",va="center",weight="bold",fontsize=12,color=NAVY)
        if sub:
            ax.text(x+w/2,y+h*.27,sub,ha="center",va="center",fontsize=9.2,color=MUTED)

    def arrow(x1,y1,x2,y2,color="#698499",label=None,offset=(0,.12)):
        a=FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,
                          color=color,linewidth=1.9,connectionstyle="arc3,rad=0")
        ax.add_patch(a)
        if label:ax.text((x1+x2)/2+offset[0],(y1+y2)/2+offset[1],label,fontsize=9,color=color,ha="center")

    ax.text(.22,6.12,"APML + MES  |  Objective architecture",fontsize=17,weight="bold",color=NAVY)
    box(.25,3.95,1.75,1.00,"Input x","label y")
    box(2.55,3.95,1.95,1.00,"Encoder","fθ(x)")
    box(5.05,3.95,1.95,1.00,"Embedding z","in Rᵈ")
    box(8.04,3.95,2.30,1.00,"Cauchy scores","a = 1/(1+||z-w||²)",face="#EAF4F4",edge="#9BD5D1")
    box(11.05,3.95,2.50,1.00,"APML","-mean log p(y|z,W)",face="#EAF4F4",edge="#9BD5D1")
    arrow(2.06,4.45,2.47,4.45);arrow(4.55,4.45,4.97,4.45);arrow(7.05,4.45,7.96,4.45);arrow(10.40,4.45,10.97,4.45)

    box(5.02,1.70,2.15,1.02,"Class proxies W","trainable and shared",face="#F1F2FB",edge="#C8C3EA")
    box(8.10,1.70,2.20,1.02,"Squared distances","qⱼₖ = ||wⱼ − wₖ||²",face="#FFF6E9",edge="#EDCC95")
    box(11.07,1.70,2.44,1.02,"MES","population Var(qⱼₖ)",face="#FFF6E9",edge="#EDCC95")
    arrow(7.21,2.20,8.02,2.20,color=ORANGE)
    arrow(10.37,2.20,10.99,2.20,color=ORANGE)
    arrow(6.10,2.79,8.93,3.85,color=LILAC,label="same W",offset=(0,.18))

    box(4.10,.20,5.74,.76,"Total loss = α · APML + β · MES","",face="#EFF3F7",edge="#ACBAC8")
    # Route APML around the MES box, not across it.
    arrow(13.64,4.45,13.87,4.45,color=TEAL)
    arrow(13.87,4.45,13.87,1.23,color=TEAL)
    arrow(13.87,1.23,9.86,.88,color=TEAL)
    arrow(11.10,1.68,9.43,1.00,color=ORANGE)
    ax.text(.28,.28,"CONCEPTUAL / NOT TO SCALE",fontsize=8.6,color=MUTED)
    save(fig,out,"02_objective_pipeline")


def tetrahedron():
    return np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],dtype=np.float64)


def octahedron():
    return np.vstack([np.eye(3),-np.eye(3)])


def draw_vertices(ax, w, *, draw_all=True, octa=False):
    n=len(w)
    for i in range(n):
        for j in range(i+1,n):
            q=float(((w[i]-w[j])**2).sum())
            if octa and q>2.5:
                ax.plot(*np.array([w[i],w[j]]).T,color=ORANGE,linewidth=1.55,linestyle="--",alpha=.85)
            else:
                ax.plot(*np.array([w[i],w[j]]).T,color=TEAL if not octa else "#AFC4CF",linewidth=2 if not octa else 1.1,alpha=.85)
    for i,p in enumerate(w):
        ax.scatter(*p,color=COLORS[i],s=80,depthshade=False,edgecolor=NAVY,linewidth=.6)
        ax.text(*(p+np.array([.13,.12,.12])),f"P{i+1}",color=NAVY,fontsize=9)
    equal_3d(ax, np.zeros(3),1.56)


def figure_03(out):
    w=tetrahedron()
    fig=plt.figure(figsize=(9.3,6.3))
    ax=fig.add_subplot(111,projection="3d")
    draw_vertices(ax,w)
    fig.suptitle("Regular tetrahedron · four equidistant proxies in 3D",fontsize=16,weight="bold",y=.96)
    fig.text(.5,.06,"ANALYTIC  •  All 6 squared distances qⱼₖ = 8  •  MES = 0  •  Not learned class weights",
             color=MUTED,fontsize=10,ha="center")
    save(fig,out,"03_ideal_tetrahedron")


def unique_q(w):
    pairs=[(i,j) for i in range(len(w)) for j in range(i+1,len(w))]
    return pairs,np.array([np.dot(w[i]-w[j],w[i]-w[j]) for i,j in pairs])


def figure_04(out,t):
    fig,ax=plt.subplots(figsize=(11.4,5.3))
    pairs, q0=unique_q(t["results"]["APML"]["weights"])
    _,q1=unique_q(t["results"]["APML + MES"]["weights"])
    xs=np.arange(len(pairs))
    ax.scatter(xs-.12,q0,s=83,color=COLORS[0],zorder=4,label="APML only")
    ax.scatter(xs+.12,q1,s=83,color=TEAL,zorder=4,label="APML + MES")
    for i,(a,b) in enumerate(zip(q0,q1)):
        ax.plot([i-.12,i+.12],[a,b],color="#B5C2CD",linewidth=1.25,zorder=2)
    ax.axhline(q0.mean(),color=COLORS[0],linewidth=1.05,linestyle=":",alpha=.7)
    ax.axhline(q1.mean(),color=TEAL,linewidth=1.05,linestyle=":",alpha=.7)
    ax.set_xticks(xs,[f"P{i+1}–P{j+1}" for i,j in pairs])
    ax.set_ylabel("Squared proxy distance  qⱼₖ",labelpad=12)
    ax.set_xlim(-.5,5.5)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y",color=LIGHT,zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(frameon=False,ncol=2,loc="upper left")
    fig.suptitle("Six unique pair distances · matched synthetic proxy experiment",fontsize=16,weight="bold",y=.98)
    fig.text(.5,.015,f"SYNTHETIC / COMPUTED  •  MES: {np.mean((q0-q0.mean())**2):.3f} (APML) vs {np.mean((q1-q1.mean())**2):.3f} (APML + MES)  •  No real-data claim",
             fontsize=9.3,ha="center",color=MUTED)
    fig.subplots_adjust(top=.84,bottom=.20)
    save(fig,out,"04_six_squared_distances")


def figure_05(out):
    fig=plt.figure(figsize=(12.5,6.5))
    tet=tetrahedron(); octa=octahedron()
    ax1=fig.add_subplot(1,2,1,projection="3d")
    draw_vertices(ax1,tet)
    ax1.set_title("C = 4, d = 3  |  equidistant possible",pad=10,fontsize=11.5,weight="bold")
    ax2=fig.add_subplot(1,2,2,projection="3d")
    draw_vertices(ax2,octa,octa=True)
    ax2.set_title("C = 6, d = 3  |  equidistant impossible",pad=10,fontsize=11.5,weight="bold")
    fig.suptitle("Dimensionality limits exact equidistance",fontsize=17,weight="bold",y=.975)
    fig.text(.5,.064,"ANALYTIC GEOMETRY  •  Tetrahedron: q = 8 for all pairs  •  Octahedron: q = 2 or 4 (example only)",
             fontsize=9.6,color=MUTED,ha="center")
    fig.text(.5,.027,"The octahedron is not claimed to minimize MES or maximize any packing criterion.",
             fontsize=9.1,color=MUTED,ha="center")
    save(fig,out,"05_dimensional_limit")


def main():
    argp=argparse.ArgumentParser(description=__doc__)
    argp.add_argument("--out",type=Path,default=Path("figures"))
    args=argp.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    apply_style()
    toy=toy_fixed_embedding_experiment()
    figure_01(args.out,toy)
    figure_02(args.out)
    figure_03(args.out)
    figure_04(args.out)
    figure_05(args.out)
    summary={
        "status":"synthetic_or_analytic_only",
        "machine":{"python":platform.python_version(),"numpy":np.__version__,
                   "torch":torch.__version__,"matplotlib":matplotlib.__version__},
        "toy_figure_01_and_04":{
            "seed":toy["seed"],"n_per_class":toy["n_per_class"],"dimensions":3,
            "class_count":4,"optimizer":"Adam","optimize":"class_proxies_only; embeddings fixed",
            "lr":toy["learning_rate"],"steps":toy["steps"],"beta":toy["beta"],
            "alpha":1.0,"provenance":"all-synthetic; no pretrained encoder; no experimental result",
            "outcomes":{},
        },
        "figure_02":{"kind":"conceptual","not_experimental":True},
        "figure_03":{"kind":"analytic","points":tetrahedron().tolist(),
                     "every_squared_pair_distance":8.0,"mes":0.0},
        "figure_05":{"kind":"analytic","C4":"tetrahedron, every q=8",
                     "C6":"octahedron, q=2 or q=4; NOT an MES optimum claim"},
    }
    for k,r in toy["results"].items():
        pairs,q=unique_q(r["weights"])
        summary["toy_figure_01_and_04"]["outcomes"][k]={
            "weights":np.round(r["weights"],12).tolist(),
            "apml":r["apml"],"mes":r["mes"],
            "pair_ids":[[i+1,j+1] for i,j in pairs],
            "squared_pair_distances":q.tolist(),
        }
    (args.out/"manifest.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("Generated five SVGs, five PNGs, and manifest.json in",args.out)
    print("Toy APML-only MES:",toy["results"]["APML"]["mes"])
    print("Toy APML+MES MES:",toy["results"]["APML + MES"]["mes"])


if __name__=="__main__":
    main()
