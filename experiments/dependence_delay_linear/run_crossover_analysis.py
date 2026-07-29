"""Post-baseline analysis of transient-versus-steady-state participation.

This analysis is intentionally labeled exploratory.  EXP-001 was evaluated
before this script was introduced and its failed interior-optimum gate remains
unchanged.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from linear_model import ModelConfig, exact_risk, make_agent_delays


AGENT_COUNTS = [1, 2, 4, 8, 16, 32]
HORIZONS = [25, 50, 100, 200, 500, 1000, 2000]
ALIGNMENTS = ["sample_time", "server_time"]
ETA = 0.02
RHO = 0.9
MAX_DELAY = 16
DELAY_EXPONENT = 1.25


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "results" / "crossover"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_delays = make_agent_delays(
        max_agents=max(AGENT_COUNTS),
        max_delay=MAX_DELAY,
        exponent=DELAY_EXPONENT,
    )

    rows = []
    for alignment in ALIGNMENTS:
        for horizon in HORIZONS:
            config = ModelConfig(
                horizon=horizon,
                common_noise_alignment=alignment,
            )
            for num_agents in AGENT_COUNTS:
                metrics = exact_risk(
                    eta=ETA,
                    rho=RHO,
                    num_agents=num_agents,
                    delays=all_delays[:num_agents],
                    config=config,
                )
                rows.append(
                    {
                        "alignment": alignment,
                        "horizon": horizon,
                        "num_agents": num_agents,
                        "eta": ETA,
                        "rho": RHO,
                        "mean_delay": float(
                            np.mean(all_delays[:num_agents])
                        ),
                        "max_delay": int(np.max(all_delays[:num_agents])),
                        **metrics,
                    }
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "crossover.csv", index=False)

    optimal_rows = []
    for (alignment, horizon), subset in frame.groupby(
        ["alignment", "horizon"]
    ):
        choice = subset.loc[subset["finite_mse"].idxmin()]
        optimal_rows.append(
            {
                "alignment": alignment,
                "horizon": int(horizon),
                "optimal_num_agents": int(choice["num_agents"]),
                "finite_mse": float(choice["finite_mse"]),
            }
        )
    optimal = pd.DataFrame(optimal_rows)
    optimal.to_csv(output_dir / "optimal_participation_by_horizon.csv", index=False)

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), sharey=True)
    for axis, alignment in zip(axes, ALIGNMENTS):
        subset_alignment = frame[frame["alignment"] == alignment]
        for horizon in HORIZONS:
            subset = subset_alignment[subset_alignment["horizon"] == horizon]
            axis.plot(
                subset["num_agents"],
                subset["finite_mse"],
                marker="o",
                label="K={0}".format(horizon),
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(AGENT_COUNTS)
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.set_xlabel("Accepted agents (fastest first)")
        axis.set_title(alignment.replace("_", " "))
    axes[0].set_ylabel("Finite-horizon MSE")
    axes[1].legend(ncol=2)
    fig.suptitle(
        r"Transient-to-steady-state crossover at $\eta=0.02,\ \rho=0.9$"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "fig_crossover_by_horizon.png")
    plt.close(fig)

    server = frame[frame["alignment"] == "server_time"]
    stationary_by_q = (
        server.groupby("num_agents", as_index=False)["stationary_mse"].first()
    )
    stationary_choice = stationary_by_q.loc[
        stationary_by_q["stationary_mse"].idxmin()
    ]
    all_agents_stationary = stationary_by_q[
        stationary_by_q["num_agents"] == max(AGENT_COUNTS)
    ].iloc[0]
    summary = {
        "experiment_id": "EXP-003-transient-stationary-crossover",
        "status": "COMPLETED_EXPLORATORY",
        "fixed_parameters": {
            "eta": ETA,
            "rho": RHO,
            "max_delay": MAX_DELAY,
            "horizons": HORIZONS,
        },
        "server_time_stationary": {
            "optimal_num_agents": int(stationary_choice["num_agents"]),
            "optimal_mse": float(stationary_choice["stationary_mse"]),
            "all_agents_mse": float(all_agents_stationary["stationary_mse"]),
            "all_agents_to_optimal_ratio": float(
                all_agents_stationary["stationary_mse"]
                / stationary_choice["stationary_mse"]
            ),
        },
        "optimal_participation_by_horizon": optimal.to_dict(orient="records"),
        "interpretation": (
            "Post-hoc mechanism analysis; it does not alter the EXP-001 "
            "pre-registered verdict."
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
