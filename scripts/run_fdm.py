"""Script for running and plotting the FDM solution."""

from viz import create_animation, plot_snapshots

from project import ( #Importerer funksjonene, som jeg har definert i fdm.py
    load_config,
    solve_heat_equation,
    solve_heat_equation_smart, 
    solve_heat_equation_optimal,
)


def main(): #Kjører og lager animasjon for FDM
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

######## Smart Varmekilde ################################################################################################
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
#############################################################################################################################

####### Funksjon som finner optimal plassering for sekundær varmekilde, med "Peis" i (0,0)########################################
def find_optimal():
    cfg = load_config("config.yaml")

    print("Finding optimal secondary position")
    x_optimal,y_optimal = solve_heat_equation_optimal(cfg) #Returnerer optimale x og y verdier for plasseringen av varmekilde 2

    print("Optimal X = ", x_optimal, ", og Optimal Y = ", y_optimal)
##################################################################################################################################

if __name__ == "__main__":
    main() #Vanlig kjøring av FDM
    main_smart() #Kjøring av samrt varmekilde som tilpasser seg 20C
    # find_optimal(), finner optimal plassering av sekundær varmekilde med "peis" i (0,0) (Denne tar veldig lenge å kjøre)
