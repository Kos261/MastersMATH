import numpy as np
import os
from itertools import combinations
import matplotlib.pyplot as plt
from initial_conditions import *

import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.animation import FuncAnimation

from poliastro.core.elements import rv_pqw, rv2coe
from poliastro.twobody import Orbit

class EquationsCore:
    def __init__(self) -> None:
        pass
    
    def eq_of_motion(self, t, states):
        """Równania ruchu z uwzględnieniem J2"""
        velocities = states[:, 3:6]  # shape (N, 3)
        accelerations = self.dynamics(states)  # shape (N, 3)
        derivatives = np.hstack([velocities, accelerations])  # shape (N, 6)
        return derivatives

    def dynamics(self, states):
        positions = states[:, :3]  # shape (N, 3)
        velocities = states[:, 3:6]  # shape (N, 3)
        r_norms = np.linalg.norm(positions, axis=1)  # shape (N,)
        
        # Two-body acceleration
        accel_2body = -MU * positions / r_norms[:, np.newaxis]**3  # shape (N, 3)
        
        return accel_2body

    def hamiltonian(self, states):
        pass


    @staticmethod
    def to_mothership_ref_frame(states):
        """
        Transformuje pozycje satelitów do układu odniesienia statku matki (LVLH).
        Zakłada, że:
            - states ma shape (T, 4, 6), gdzie 4 to liczba satelitów
            - statek-matka to satelita o indeksie 0
        Zwraca pozycje WSZYSTKICH satelitów (w tym statku-matki) w układzie LVLH: shape (T, 4, 3)
        """

        T = states.shape[0]
        rel_pos_lvlh = np.zeros((T, 4, 3))  # (czas, satelita, xyz)

        for t in range(T):
            r_ref = states[t, 0, 0:3]
            v_ref = states[t, 0, 3:6]

            # LVLH axes at time t
            z_l = -r_ref / np.linalg.norm(r_ref)
            y_temp = np.cross(r_ref, v_ref)
            y_l = y_temp / np.linalg.norm(y_temp)
            x_l = np.cross(y_l, z_l)

            # Rotation matrix: inertial → LVLH
            R = np.vstack([x_l, y_l, z_l]).T  # shape (3, 3)

            # statek matka -> punkt (0,0,0)
            rel_pos_lvlh[t, 0] = np.zeros(3)

            # pozostałe 3 satelity
            rel_global = states[t, 1:4, 0:3] - r_ref  # shape (3, 3)
            rel_local = rel_global @ R               # shape (3, 3)
            rel_pos_lvlh[t, 1:4] = rel_local

        return rel_pos_lvlh

class Plotter:
    def __init__(self):
        self.anim = None
        # fig = plt.figure(figsize=(14, 8))
        # outer_gs = GridSpec(2, 2, width_ratios=[2.5, 1], height_ratios=[2, 1], figure=fig)

        # # Duży 3D
        # self.ax_big_3d = fig.add_subplot(outer_gs[:, :1], projection='3d')
        # self.ax_big_3d.set_title("Duży wykres 3D")

        # # Mały 3D
        # self.ax_small_3d = fig.add_subplot(outer_gs[0, 1], projection='3d')
        # self.ax_small_3d.set_title("Mały wykres 3D")

        # # Zagnieżdżony GridSpec (2x2) wewnątrz prawego dolnego bloku
        # self.inner_gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[1, 1])
        
        # self.ax_err_1 = fig.add_subplot(self.inner_gs[0, 0])
        # # self.ax_err_1.set_title("Błąd 1")

        # self.ax_err_2 = fig.add_subplot(self.inner_gs[0, 1])
        # # self.ax_err_2.set_title("Błąd 2")

        # self.ax_err_3 = fig.add_subplot(self.inner_gs[1, 0])
        # # self.ax_err_3.set_title("Błąd 3")

        # self.ax_err_4 = fig.add_subplot(self.inner_gs[1, 1])
        # # self.ax_err_4.set_title("Błąd 4")
        # self.axes_errors = [self.ax_err_1, self.ax_err_2, self.ax_err_3, self.ax_err_4]

    def plot_panel(self, times, states):

        self.plot_orbit_3d(times, states)
        self.animate_formation(states)
        self.plot_errors(times, states)

        plt.tight_layout()
        plt.show()

    def plot_orbit_3d(self, times, states):
        fig = plt.figure(figsize=(8, 8))
        self.ax_big_3d = fig.add_subplot(111, projection='3d')
        
        u = np.linspace(0, 2*np.pi, 10)
        v = np.linspace(0, np.pi, 10)
        x = R_EARTH * np.outer(np.cos(u), np.sin(v))
        y = R_EARTH * np.outer(np.sin(u), np.sin(v))
        z = R_EARTH * np.outer(np.ones(np.size(u)), np.cos(v))
        self.ax_big_3d.plot_surface(x, y, z, color='blue', alpha=0.5)
        
        colors = ['red', 'green', 'blue', 'yellow']
        labels = ['Mothership', 'Sat1', 'Sat2', 'Sat3']
        step = 50
        for i in range(4):
            self.ax_big_3d.plot(states[::step, i, 0], 
                    states[::step, i, 1], 
                    states[::step, i, 2], 
                    color=colors[i], 
                    label=labels[i], 
                    linewidth=1.5)
        
        # Punkty startowe
        self.ax_big_3d.scatter(states[0, :, 0], states[0, :, 1], states[0, :, 2], 
                c='black', s=50, marker='o', depthshade=False, label='Start')
        
        # Konfiguracja wykresu
        self.ax_big_3d.set_xlabel('X [km]', fontsize=10)
        self.ax_big_3d.set_ylabel('Y [km]', fontsize=10)
        self.ax_big_3d.set_zlabel('Z [km]', fontsize=10)
        self.ax_big_3d.set_title('Symulacja formacji satelitów (RK4)', fontweight='bold')
        
        # Ustawienie jednakowej skali osi
        max_range = np.max(np.abs(states[:, :, :3])) * 1.2
        self.ax_big_3d.set_xlim([-max_range, max_range])
        self.ax_big_3d.set_ylim([-max_range, max_range])
        self.ax_big_3d.set_zlim([-max_range, max_range])
        
        # Legenda i siatka
        self.ax_big_3d.legend(loc='upper right', fontsize=8)
        self.ax_big_3d.grid(True, linestyle=':', alpha=0.5)
        
        line_step = 1000
        for t_idx in range(0, len(times), line_step):
            positions = states[t_idx]  # shape (4, 3)
            for i, j in combinations(range(4), 2):
                x = [positions[i, 0], positions[j, 0]]
                y = [positions[i, 1], positions[j, 1]]
                z = [positions[i, 2], positions[j, 2]]
                self.ax_big_3d.plot(x, y, z, color='black', linewidth=0.8, alpha=0.5)

        plt.tight_layout()
        plt.show()

    def plot_orbit_3d_plotly(self, times, states, draw_formation = False):
        colors = ['red', 'green', 'blue', 'yellow']
        labels = ['Mothership', 'Sat1', 'Sat2', 'Sat3']
        step = 50

        fig = go.Figure()

        u = np.linspace(0, 2 * np.pi, 15)
        v = np.linspace(0, np.pi, 15)
        ex = R_EARTH * np.outer(np.cos(u), np.sin(v))
        ey = R_EARTH * np.outer(np.sin(u), np.sin(v))
        ez = R_EARTH * np.outer(np.ones(np.size(u)), np.cos(v))

        fig.add_trace(go.Surface(
            x=ex, y=ey, z=ez,
            colorscale='Blues',
            opacity=0.5,
            showscale=False,
            name='Earth',
            hoverinfo='skip'
        ))

        for i in range(4):
            fig.add_trace(go.Scatter3d(
                x=states[::step, i, 0],
                y=states[::step, i, 1],
                z=states[::step, i, 2],
                mode='lines',
                line=dict(color=colors[i], width=3),
                name=labels[i]
            ))

        fig.add_trace(go.Scatter3d(
            x=states[0, :, 0],
            y=states[0, :, 1],
            z=states[0, :, 2],
            mode='markers',
            marker=dict(color='black', size=5),
            name='Start'
        ))

        if draw_formation:
            line_step = 1000
            for t_idx in range(0, len(times), line_step):
                positions = states[t_idx]
                for i, j in combinations(range(4), 2):
                    fig.add_trace(go.Scatter3d(
                        x=[positions[i, 0], positions[j, 0]],
                        y=[positions[i, 1], positions[j, 1]],
                        z=[positions[i, 2], positions[j, 2]],
                        mode='lines',
                        line=dict(color='black', width=1, dash='dot'),
                        showlegend=False
                    ))

        max_range = np.max(np.abs(states[:, :, :3])) * 1.2
        fig.update_layout(
            scene=dict(
                xaxis_title='X [km]',
                yaxis_title='Y [km]',
                zaxis_title='Z [km]',
                xaxis=dict(range=[-max_range, max_range]),
                yaxis=dict(range=[-max_range, max_range]),
                zaxis=dict(range=[-max_range, max_range]),
                aspectmode='cube',
            ),
            title='Symulacja formacji satelitów (RK4)',
            legend=dict(x=0.02, y=0.98)
        )

        fig.show()

    def animate_formation(self, states, step=50, lvlh=False):
        # T = states.shape[0]
        if lvlh:
            states = EquationsCore.to_mothership_ref_frame(states)


        fig = plt.figure(figsize=(8, 8))
        self.ax_small_3d = fig.add_subplot(111, projection='3d')
        self.ax_small_3d.set_xlim([-400, 400])
        self.ax_small_3d.set_ylim([-400, 400])
        self.ax_small_3d.set_zlim([-400, 400])
        self.ax_small_3d.set_xlabel("X [km] (prostopadle do orbity)")
        self.ax_small_3d.set_ylabel("Y [km] (wzdłuż orbity)")
        self.ax_small_3d.set_zlabel("Z [km] (do Ziemi)")
        self.ax_small_3d.set_title("Formacja satelitów w układzie LVLH")
        self.ax_small_3d.scatter([0], [0], [0], color='red', label='Mothership', s=30)
        self.ax_small_3d.legend()

        # początkowe linie między satelitami
        lines = []
        for (i,j) in combinations(range(4), 2):
            line, = self.ax_small_3d.plot([0], [j], [i], color='black', linewidth=1)
            lines.append(line)

        pos = states[0,:,0:3]
        ref = pos[0,0:3]  # pozycja statku-matki
        rel_pos = pos - ref

        # for (i, j) in combinations(range(3), 2):
        #     x = [rel_pos[i, 0], rel_pos[j, 0]]
        #     y = [rel_pos[i, 1], rel_pos[j, 1]]
        #     z = [rel_pos[i, 2], rel_pos[j, 2]]
        #     lines[0].set_data(x, y)
        #     lines[0].set_3d_properties(z)

        def update(frame):
            t_idx = frame * step
            if t_idx >= len(states):
                return lines

            pos = states[t_idx,:,0:3]
            ref = pos[0,0:3]  # pozycja statku-matki
            rel_pos = pos - ref  # przesunięcie względem statku-matki

            for k, (i, j) in enumerate(combinations(range(4), 2)):
                x = [rel_pos[i, 0], rel_pos[j, 0]]
                y = [rel_pos[i, 1], rel_pos[j, 1]]
                z = [rel_pos[i, 2], rel_pos[j, 2]]
                lines[k].set_data(x, y)
                lines[k].set_3d_properties(z)
            return lines

        frames = max(1, len(states) // step)
        self.anim = FuncAnimation(fig, update, frames=frames, blit=False, interval=100)
        plt.tight_layout()
        plt.show()
 
    def plot_errors(self, times, states):
        positions = states[:, :, :3]  # shape (n_steps, 4, 3)
        n_steps = len(times)
        
        dist_errors = np.zeros((n_steps, 4))  # 4 satellite pairs
        
        dist_errors[:, 0] = np.linalg.norm(positions[:, 0] - positions[:, 1], axis=1) - sat_separation
        dist_errors[:, 1] = np.linalg.norm(positions[:, 1] - positions[:, 2], axis=1) - sat_separation
        dist_errors[:, 2] = np.linalg.norm(positions[:, 2] - positions[:, 3], axis=1) - sat_separation
        dist_errors[:, 3] = np.linalg.norm(positions[:, 3] - positions[:, 0], axis=1) - sat_separation
        
        #Plot settings
        plt.figure(figsize=(10, 8))
        plt.suptitle(f"Satellite Formation Distance Errors (Target: {sat_separation} km)", y=1.02)
        
        pair_names = ['Sat1-Sat2', 'Sat2-Sat3', 'Sat3-Sat4', 'Sat4-Sat1']
        colors = ['b', 'g', 'r', 'm']


        # for i in range(4):
        #     plt.subplot(4, 1, i+1)
        #     plt.plot(times, dist_errors[:, i], color=colors[i], label=pair_names[i])
        #     plt.ylabel('Error (m)')
        #     plt.grid(True, alpha=0.3)
        #     plt.legend(loc='upper right')
            
        #     # Add horizontal line at zero for reference
        #     plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
            
        #     if i == 3:
        #         plt.xlabel('Time (s)')
        #     else:
        #         plt.tick_params(labelbottom=False)

        # plt.tight_layout()
        # plt.show()


        for i in range(3):
            plt.subplot(3, 1, i+1)
            plt.plot(times, positions[:, 1, i], color=colors[i], label=pair_names[i])
            plt.ylabel('Position')
            plt.xlabel('Time(s)')
            plt.grid(True, alpha=0.3)
            

        plt.tight_layout()
        plt.show()


def runge_kutta_4(state, t, dt, f):
    k1 = f(t, state)
    k2 = f(t + dt/2, state + dt/2 * k1)
    k3 = f(t + dt/2, state + dt/2 * k2)
    k4 = f(t + dt, state + dt * k3)
    
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def propagate_orbit_rk4(state0, t0, tf, dt, f):
    """Propagacja orbity z użyciem RK4"""


    times = np.arange(t0, tf, dt)
    states = np.zeros((len(times), 4, 6))
    states[0] = state0
    # print(states[0])
    
    for i in range(1, len(times)):
        states[i] = runge_kutta_4(states[i-1], times[i-1], dt, f)
    
    return times, states

def compute_states(filename, state0, t0, tf, dt, eq_of_motion):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, filename)
    # print("PATH: ",path)
    if os.path.exists(path):
        print(f"[cache] loading from {path}")
        data = np.load(path)
        times = data["times"]
        states = data["states"]
    else:
        print(f"[compute] computing orbit")
        times, states = propagate_orbit_rk4(state0, t0, tf, dt, eq_of_motion)
        np.savez(path, times=times, states=states)

    return times, states


if __name__ == "__main__":
 
    state0 = formation()

    core = EquationsCore()

    '''
    Orbit propagation
    States (len(time), satelitte num, x y z vx vy vz)
    Shape  (len(time),       4,             6)
    '''
    times, states = compute_states(filename, state0, t0, tf, dt, core.eq_of_motion)


    times, states = propagate_orbit_rk4(state0, t0, tf, dt, core.eq_of_motion)
    lvlh_states = core.to_mothership_ref_frame(states)



    plotter = Plotter()
    # plotter.plot_panel(times, states)
    # plotter.animate_formation(states, step=50, lvlh=True)
    # plotter.plot_errors(times, states)    
    plotter.plot_orbit_3d_plotly(times, states)