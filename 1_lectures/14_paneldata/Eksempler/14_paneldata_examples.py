"""Python examples for the panel-data lecture.

The script replaces the old Stata examples and produces the simulation
figure used in the slides.  The functions are deliberately small so they
can also be copied into an interactive notebook.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path


# Run this file from the Eksempler folder, just like the notebook.
DATA = Path("../../data")
OUT = Path(".")


def simulate_panel(n=500, t=2, beta=2.0, sigma_a=2.0, sigma_v=2.0,
                   sigma_u=1.0, loading_a=1.0, seed=117):
    """Simulate y=1+beta*x+a+u with x=2+v+loading_a*a."""
    rng = np.random.default_rng(seed)
    i = np.repeat(np.arange(n), t)
    time = np.tile(np.arange(1, t + 1), n)
    a = np.repeat(rng.normal(0, sigma_a, n), t)
    v = rng.normal(0, sigma_v, n * t)
    u = rng.normal(0, sigma_u, n * t)
    x = 2 + v + loading_a * a
    y = 1 + beta * x + a + u
    return pd.DataFrame({"i": i, "t": time, "x": x, "y": y})


def estimate_panel(data):
    """OLS, FD, FE and feasible RE, including common time effects.

    The balanced-panel RE transformation estimates variance components
    from within and between residuals (Swamy-Arora). Standard errors in
    this table are conventional model-based SE, not clustered SE.
    They do not make an inconsistent estimator valid.
    """
    d = data.sort_values(["i", "t"]).copy()
    counts = d.groupby("i").size()
    n, t = len(counts), int(counts.iloc[0])
    if not counts.eq(t).all() or t < 2:
        raise ValueError("Use a balanced panel with at least two periods.")
    if d.duplicated(["i", "t"]).any() or d.groupby("i").t.nunique().ne(t).any():
        raise ValueError("Each individual must have one observation per period.")
    if d.groupby("t").size().ne(n).any():
        raise ValueError("All individuals must be observed in the same periods.")

    time = pd.get_dummies(d.t, prefix="time", drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([d[["x"]], time], axis=1))
    pooled = sm.OLS(d.y, X).fit()

    first = d.groupby("i")[["y", "x"]].diff().dropna()
    fd_time = pd.get_dummies(d.loc[first.index, "t"], prefix="time",
                             drop_first=True, dtype=float)
    fd_X = sm.add_constant(pd.concat([first[["x"]], fd_time], axis=1))
    fd = sm.OLS(first.y, fd_X).fit()

    # Within transformation: retain the transformed time dummies.
    y_within = d.y - d.groupby("i").y.transform("mean")
    X_within = X.drop(columns="const")
    X_within = X_within - X_within.groupby(d.i).transform("mean")
    fe = sm.OLS(y_within, X_within).fit()
    df_within = n * (t - 1) - X_within.shape[1]
    sigma_u2 = fe.ssr / df_within
    # n individual intercepts were removed: nT observations are not nT
    # independent within observations.
    fe_se = np.sqrt(sigma_u2 * fe.normalized_cov_params.loc["x", "x"])

    means = d.groupby("i")[["y", "x"]].mean()
    between = sm.OLS(means.y, sm.add_constant(means.x)).fit()
    sigma_a2 = max(0.0, between.mse_resid - sigma_u2 / t)
    theta = 1 - np.sqrt(sigma_u2 / (sigma_u2 + t * sigma_a2))
    y_re = d.y - theta * d.groupby("i").y.transform("mean")
    X_re = X - theta * X.groupby(d.i).transform("mean")
    re = sm.OLS(y_re, X_re).fit()

    table = pd.DataFrame({
        "estimate": [pooled.params["x"], fd.params["x"],
                     fe.params["x"], re.params["x"]],
        "std_error": [pooled.bse["x"], fd.bse["x"], fe_se, re.bse["x"]]
    }, index=["OLS", "FD", "FE", "RE"])
    table.attrs["theta"] = theta
    return table


def simulation_results(reps=1000, **kwargs):
    """Monte Carlo distribution of the four estimators."""
    return pd.DataFrame([estimate_panel(simulate_panel(seed=117 + r, **kwargs))["estimate"]
                         for r in range(reps)])


def plot_panel_transformations(data, path=OUT / "PanelTransformations.pdf"):
    """Compare pooled OLS, FD and within on the same small panel.

    No time effects are included here. In the raw-data panel the FD and
    FE slopes pass through the grand mean solely as a visual reference:
    the transformations do not identify a common intercept in levels.
    """
    d = data.sort_values(["i", "t"]).copy()
    fd = d.groupby("i")[["x", "y"]].diff().dropna()
    within = d[["x", "y"]] - d.groupby("i")[["x", "y"]].transform("mean")
    pooled_fit = sm.OLS(d.y, sm.add_constant(d.x)).fit()
    fd_fit = sm.OLS(fd.y, sm.add_constant(fd.x)).fit()
    fe_fit = sm.OLS(within.y, within[["x"]]).fit()
    slopes = {"OLS": pooled_fit.params["x"],
              "FD": fd_fit.params["x"], "FE": fe_fit.params["x"]}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    colours = plt.get_cmap("tab10")
    for k, (individual, rows) in enumerate(d.groupby("i")):
        colour = colours(k % 10)
        for ax, values in zip(axes, [d, fd, within]):
            points = values.loc[values.index.intersection(rows.index)]
            ax.scatter(points.x, points.y, color=colour, s=32, alpha=.85,
                       edgecolors="white", linewidths=.35,
                       label=f"Individ {individual + 1}")

    grid = np.linspace(d.x.min() - .3, d.x.max() + .3, 150)
    axes[0].plot(grid, pooled_fit.params["const"] + slopes["OLS"] * grid,
                 color="black", lw=2.2, label=f"OLS: {slopes['OLS']:.2f}")
    for method, colour, style in [("FD", "#c62828", "--"),
                                   ("FE", "#176b3a", ":")]:
        axes[0].plot(grid, d.y.mean() + slopes[method] * (grid - d.x.mean()),
                     color=colour, ls=style, lw=2.6,
                     label=f"{method}: {slopes[method]:.2f}")
    for ax, values, fit, colour, method in [
        (axes[1], fd, fd_fit, "#c62828", "FD"),
        (axes[2], within, fe_fit, "#176b3a", "FE")
    ]:
        grid = np.linspace(values.x.min() - .2, values.x.max() + .2, 150)
        ax.plot(grid, fit.params.get("const", 0) + fit.params["x"] * grid,
                color=colour, lw=2.2, label=f"OLS: {slopes[method]:.2f}")
        ax.axhline(0, color=".8", lw=.7, zorder=0)
        ax.axvline(0, color=".8", lw=.7, zorder=0)
        ax.legend(handles=[ax.lines[0]], loc="upper left", frameon=False,
                  fontsize=9)
    for ax, title, xlabel, ylabel in zip(
        axes, ["Rå data", "Første differenser (FD)", "Within (FE)"],
        [r"$x_{it}$", r"$\Delta x_{it}$", r"$\ddot{x}_{it}$"],
        [r"$y_{it}$", r"$\Delta y_{it}$", r"$\ddot{y}_{it}$"]
    ):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[-3:], labels[-3:], loc="upper left",
                   frameon=False, fontsize=10)
    fig.legend(handles[:-3], labels[:-3], loc="lower center", ncol=6,
               frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, .10, 1, 1), w_pad=2)
    if path is not None:
        fig.savefig(path, bbox_inches="tight")
    return fig, slopes


def plot_simulation(results, path=OUT / "PanelSim.pdf", beta=2.0):
    """Plot OLS and FD sampling distributions for the slides."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(results.OLS, bins=35, alpha=.65, color="crimson", label="OLS")
    ax.hist(results.FD, bins=35, alpha=.65, color="navy", label="FD")
    ax.axvline(beta, color="black", linestyle="--", label=rf"Sand værdi $\beta_1={beta:g}$")
    ax.set_xlabel(r"$\hat\beta_1$")
    ax.set_ylabel("Frekvens")
    ax.set_title("Paneldata: pooled OLS og første differenser")
    ax.legend(frameon=False)
    fig.tight_layout()
    if path is None:
        plt.show()
    else:
        fig.savefig(path)
    plt.close(fig)


def difference_in_differences(cov_type="nonrobust"):
    """Reproduce the Kielmc DiD regression in Python."""
    d = pd.read_stata(DATA / "kielmc.dta")
    d["did"] = d.y81 * d.nearinc
    X = sm.add_constant(d[["y81", "nearinc", "did"]])
    # Levels, in thousands of 1978 dollars, as in the lecture table.
    # The conventional SE reproduce Wooldridge's table; try HC1 to compare.
    return sm.OLS(d.rprice / 1000, X).fit(cov_type=cov_type)


def police_difference_in_differences():
    """Reproduce the police-protection DiD and FD regression in Python."""
    d = pd.read_stata(DATA / "police.dta")
    d = d.loc[~d.distanci.isin([1, 2])].copy()
    d["post"] = np.select([d.mes.between(4, 6), d.mes.between(8, 10)], [0, 1], default=np.nan)
    d = d.dropna(subset=["post"])
    d = d.groupby(["observ", "post"], as_index=False).agg(totrob=("totrob", "mean"),
                                                             institu1=("institu1", "first"))
    d["post_institu1"] = d.post * d.institu1
    X = sm.add_constant(d[["institu1", "post", "post_institu1"]])
    did = sm.OLS(d.totrob, X).fit(cov_type="HC1")
    ordered = d.sort_values(["observ", "post"])
    fd = ordered.groupby("observ").totrob.diff().dropna().to_frame("d_totrob")
    treatment = ordered.groupby("observ").institu1.first()
    fd["institu1"] = ordered.loc[fd.index, "observ"].map(treatment).to_numpy()
    fd_fit = sm.OLS(fd.d_totrob, sm.add_constant(fd.institu1)).fit(cov_type="HC1")
    return did, fd_fit


def export_slide_results(table, path=OUT / "PanelEstimates.tex"):
    """Write the slide table directly from the estimated coefficients and SE."""
    with open(path, "w") as file:
        file.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        file.write("Estimator & OLS & FD & FE & RE \\\\\n\\midrule\n")
        for label, column in [(r"$\hat\beta_1$", "estimate"),
                              (r"SE($\hat\beta_1$)", "std_error")]:
            values = " & ".join(f"{x:.4f}".replace(".", ",") for x in table[column])
            file.write(f"{label} & {values} \\\\\n")
        file.write("\\bottomrule\n\\end{tabular}\n")


def export_police_results(did, fd, path=OUT / "PoliceEstimates.tex"):
    """Coefficients and HC1 SE from exactly the regressions in the notebook."""
    with open(path, "w") as file:
        file.write("\\begin{center}\\small\n\\begin{tabular}{lrr}\n\\toprule\n")
        file.write("Estimation & DiD-hældning & Robust SE \\\\\n\\midrule\n")
        for label, result, term in [
            ("DiD", did, "post_institu1"), ("FD", fd, "institu1")
        ]:
            b = f"{result.params[term]:.4f}".replace(".", ",")
            se = f"{result.bse[term]:.4f}".replace(".", ",")
            file.write(label + " & $" + b + "$ & $" + se + "$ \\\\\n")
        file.write("\\bottomrule\n\\end{tabular}\\end{center}\n")


if __name__ == "__main__":
    # Small balanced panel for the raw / FD / within illustration.
    small_panel = simulate_panel(n=6, t=10, sigma_a=2.5, sigma_v=1, sigma_u=.5)
    transformation_figure, slopes = plot_panel_transformations(small_panel)
    plt.close(transformation_figure)
    print("Transformation figure:", slopes)
    estimates = estimate_panel(simulate_panel())
    print(estimates.round(4))
    export_slide_results(estimates)
    results = simulation_results()
    print(results.mean().to_frame("mean").round(3))
    plot_simulation(results)
    print("\nKielmc DiD:\n", difference_in_differences().summary())
    police_did, police_fd = police_difference_in_differences()
    export_police_results(police_did, police_fd)
