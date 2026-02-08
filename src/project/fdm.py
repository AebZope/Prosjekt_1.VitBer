"""Finite Difference Method solver for the 2D heat equation."""

import numpy as np

from .config import Config

def solve_heat_equation(
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the 2D heat equation using implicit Euler.

    Args:
        cfg: Configuration object

    Returns:
        x: x-coordinates (nx,)
        y: y-coordinates (ny,)
        t: time points (nt,)
        T: temperature solution (nt, nx, ny)
    """
    # Create grids
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    y = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)
    t = np.linspace(cfg.t_min, cfg.t_max, cfg.nt)

    dx, dy = x[1] - x[0], y[1] - y[0]
    dt = t[1] - t[0]

    X, Y = np.meshgrid(x, y, indexing="ij")

    #######################################################################
    # Oppgave 3.2: Start
    #######################################################################

    T = np.zeros((cfg.nt, cfg.nx, cfg.ny))

    # Initialbetingelse: T(x, y, 0) = temperatur utenfor rommet.
    T[0, :, :] = cfg.T_outside

    # bygger matrise A
    A = _build_matrix(cfg, dx, dy, dt)

    #itererer gjennom systemets og bygger matrise b for hvert tidssteg.
    for k in range(cfg.nt - 1):
        b = _build_rhs(cfg, T_curr=T[k], X=X, Y=Y, dx=dx, dy=dy,dt=dt, t_next=t[k + 1],varmekilde = True)

        # Løser ligningssystemet for temperatur i neste tidssteg. 
        T_next = np.linalg.solve(A, b)

        # Lagrer løsning på 2D-form
        T[k + 1] = T_next.reshape(cfg.nx, cfg.ny)

    #######################################################################
    # Oppgave 3.2: Slutt
    #######################################################################

    return x, y, t, T


######### Smart Varmekilde ########################################################################################################################
def solve_heat_equation_smart( #oppgave 3.4, forsøker å lage en smart varmekilde som skrur seg av når gjennomsnittstemperaturen overstiger en bestemt verdi.
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    
    varmekilde = True #Definerer en bool variabel som bestemmer om varmekilden er av eller på.

    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    y = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)
    t = np.linspace(cfg.t_min, cfg.t_max, cfg.nt)

    dx, dy = x[1] - x[0], y[1] - y[0]
    dt = t[1] - t[0]

    X, Y = np.meshgrid(x, y, indexing="ij")

    T = np.zeros((cfg.nt, cfg.nx, cfg.ny)) 

    # Initialbetingelse: T(x, y, 0) = temperatur utenfor rommet.
    T[0, :, :] = cfg.T_outside

    # bygger matrise A
    A = _build_matrix(cfg, dx, dy, dt)

    #itererer gjennom systemets og bygger matrise b for hvert tidssteg.
    for k in range(cfg.nt - 1):
        b = _build_rhs(cfg, T_curr=T[k], X=X, Y=Y, dx=dx, dy=dy,dt=dt, t_next=t[k + 1],varmekilde = varmekilde)
        #Jeg endret på _buold_rhs slik at den tar inn bool variablen varmekilde, slik at vi kan skru av og på varmekilden. 
        sensor_data = [] #Definerer en liste av sensordata, for temperaturen ved sensorer

        # Løser ligningssystemet for temperatur. 
        T_next = np.linalg.solve(A, b)


        for xs,ys in cfg.sensor_locations: #Itererer gjennom (x,y) posisjonen til sensorene.  
            i = np.argmin(np.abs(x-xs)) #Løser for tilsvarende indeksering i koordinatsystemet vårt. 
            j = np.argmin(np.abs(y-ys))

            sensor_data.append(float(T[k,i,j])) #Legger til temperaturen ved gitt tidssteg og indeksering til sensordata

        if np.mean(sensor_data) <= 20: #skjekker om gjennomsnittstemperatur over sensorene er under en bestemt temperatur
            varmekilde = True #Om temperatur er for lav må varmekilden skrus på. 
        else:
            varmekilde = False #Om den er for høy må varmekilden skrus av for at systemet skal avkjøles
        

     # Lagrer løsning på 2D-form
        T[k + 1] = T_next.reshape(cfg.nx, cfg.ny)
    return x, y, t, T
########################################################################################################################################


######### Optimal plassering av sekundær varmekilde ####################################################################################
def solve_heat_equation_optimal( 
    cfg: Config,
):
    
    max_temp = 0 #Variabel som vil tilsvare den høyeste gjennomsnittstemperaturen
    max_x = 0 # gir x verdi for plassering ved max_temp
    max_y = 0 # tilsvarende for y
    
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    y = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)
    t = np.linspace(cfg.t_min, cfg.t_max, cfg.nt)

    dx, dy = x[1] - x[0], y[1] - y[0]
    dt = t[1] - t[0]

    X, Y = np.meshgrid(x, y, indexing="ij")

    T = np.zeros((cfg.nt, cfg.nx, cfg.ny))

    # Initialbetingelse: T(x, y, 0) = temperatur utenfor rommet.
    T[0, :, :] = cfg.T_outside

    # bygger matrise A
    A = _build_matrix(cfg, dx, dy, dt)

    #itererer gjennom systemets og bygger matrise b for hvert tidssteg.
    for i in np.arange(0,10,0.1): #Itererer gjennom x-verdier, fra 0 til 10 med steg 0.1 
            print(int(i*10), "%:",max_x, "and", max_y) #Printer framgang underveis så jeg ser at programmet kjører 
            for j in np.arange(0,5,0.1): #Itererer gjennom y-verdier fra 0 til 5 med steg 0.1
                for k in range(cfg.nt - 1): #itererer over tidssteg og løser for T basert på (i,j) posisjon av varmekilde
                    cx = i
                    cy = j
                    b = _build_rhs_optimal(cfg, T_curr=T[k], X=X, Y=Y, dx=dx, dy=dy,dt=dt, t_next=t[k + 1],cx=cx,cy=cy)
                    #Ny variabel _build_rhs_optimal, som tar inn verdier for sekundær varmekildes plassering. 

                    # Løser ligningssystemet for temperatur. 
                    T_next = np.linalg.solve(A, b)

                    # Lagrer løsning på 2D-form
                    T[k + 1] = T_next.reshape(cfg.nx, cfg.ny)

                 
                if np.mean(T[-1]) > max_temp: #Skjekker om den endelige gjennomsnittstemperaturen til systemet er høyere enn den største før det 
                    max_temp = np.mean(np.mean(T[-1]))
                    max_x = i # Erstatter max_temp og tilsvarende max x/y med nye verdier med høyere temperatur.
                    max_y = j
            

    return max_x, max_y #Returnerer posisjonene x og y som gir høyest endelig gjennomsnittstemperatur. 
########################################################################################################################################





def _build_matrix(cfg: Config, dx: float, dy: float, dt: float) -> np.ndarray:
    """Build the implicit Euler system matrix."""
    n = cfg.nx * cfg.ny
    A = np.zeros((n, n))

    rx = cfg.alpha * dt / dx**2
    ry = cfg.alpha * dt / dy**2

    def idx(i, j):
        return i * cfg.ny + j

    I, J = np.meshgrid(np.arange(cfg.nx), np.arange(cfg.ny), indexing="ij")

    # Boundary masks
    left = I == 0
    right = I == cfg.nx - 1
    bottom = J == 0
    top = J == cfg.ny - 1

    # Diagonal entries
    diag = np.full((cfg.nx, cfg.ny), 1 + 2 * rx + 2 * ry)
    diag[left | right] -= rx
    diag[bottom | top] -= ry
    diag[left | right] += rx * cfg.h * dx / cfg.k
    diag[bottom | top] += ry * cfg.h * dy / cfg.k

    p = idx(I, J)
    A[p, p] = diag

    # Off-diagonals
    mask = ~left
    A[idx(I[mask], J[mask]), idx(I[mask] - 1, J[mask])] = -rx

    mask = ~right
    A[idx(I[mask], J[mask]), idx(I[mask] + 1, J[mask])] = -rx

    mask = ~bottom
    A[idx(I[mask], J[mask]), idx(I[mask], J[mask] - 1)] = -ry

    mask = ~top
    A[idx(I[mask], J[mask]), idx(I[mask], J[mask] + 1)] = -ry

    return A


def _build_rhs(cfg: Config, T_curr, X, Y, dx, dy, dt, t_next,varmekilde):
    """Build right-hand side for implicit system."""
    rhs = T_curr.copy()

    # Heat source (legger til en if for å skru den av og på med den smarte varmekilden.)
    if varmekilde == True:
        q = np.array(cfg.heat_source(X, Y, t_next))
    else:
        q = 0
    
    rhs += dt * q

    # Robin BC contributions
    rx = cfg.alpha * dt / dx**2
    ry = cfg.alpha * dt / dy**2
    bc_term = cfg.T_outside

    rhs[0, :] += rx * (cfg.h * dx / cfg.k) * bc_term
    rhs[-1, :] += rx * (cfg.h * dx / cfg.k) * bc_term
    rhs[:, 0] += ry * (cfg.h * dy / cfg.k) * bc_term
    rhs[:, -1] += ry * (cfg.h * dy / cfg.k) * bc_term

    return rhs.flatten()

######## Optimal posisjon #####################################################################################################
# jeg må omdefinere rhs for hver løsning av varmeligningen med forskjellig posisjon av varmekilde. 
def _build_rhs_optimal(cfg: Config, T_curr, X, Y, dx, dy, dt, t_next,cx,cy):
    """Build right-hand side for implicit system."""
    rhs = T_curr.copy()

    # Heat source (legger til en if for å skru den av og på med den smarte varmekilden.)

    q = np.array(cfg.heat_source_optimal(X, Y, t_next,cx,cy)) #Bruker ny funksjon "heat_source_optimal" definert i config.py
    
    rhs += dt * q

    # Robin BC contributions
    rx = cfg.alpha * dt / dx**2
    ry = cfg.alpha * dt / dy**2
    bc_term = cfg.T_outside

    rhs[0, :] += rx * (cfg.h * dx / cfg.k) * bc_term
    rhs[-1, :] += rx * (cfg.h * dx / cfg.k) * bc_term
    rhs[:, 0] += ry * (cfg.h * dy / cfg.k) * bc_term
    rhs[:, -1] += ry * (cfg.h * dy / cfg.k) * bc_term

    return rhs.flatten()
####################################################################################################################################