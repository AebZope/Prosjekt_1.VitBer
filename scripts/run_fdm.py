"""Script for running and plotting the FDM solution."""

from viz import create_animation, plot_snapshots

from project import (
    load_config,
    solve_heat_equation,
    solve_heat_equation_smart,
)


def main():
    cfg = load_config("config.yaml")

    print("Solving heat equation with FDM...")
    x, y, t, T_fdm = solve_heat_equation(cfg)

    print("\nGenerating FDM visualizations...")
    plot_snapshots(
        x,
        y,
        t,
        T_fdm,
        save_path="output/fdm/fdm_snapshots.png",
    )
    create_animation(
        x, y, t, T_fdm, title="FDM", save_path="output/fdm/fdm_animation.gif"
    )


def main_smart(): 
    cfg = load_config("config.yaml")

    print("Solving heat equation with FDM (Smart)...")
    x, y, t, T_fdm_smart = solve_heat_equation_smart(cfg)

    print("\nGenerating FDM(Smart) visualization...")
    plot_snapshots(
        x,
        y,
        t,
        T_fdm_smart,
        save_path="output/fdm/fdm_snapshots_smart.png",
    )
    create_animation(
        x, y, t, T_fdm_smart, title="FDM", save_path="output/fdm/fdm_animation_smart.gif"
    )


if __name__ == "__main__":
    main_smart()
