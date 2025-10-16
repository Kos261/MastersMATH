import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.style.use( 'dark_background' )
# from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.animation import FuncAnimation
import numpy as np
from itertools import combinations
from orbits import LVLH, RE


class Plotter:
    def __init__(self):
        pass 

    def plot_orbit_3d(self, times, states, orbit):
        T = states.shape[0]
        num_sat = states.shape[1]
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # draw sphere
        R = 3500.0
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = R * np.cos(u) * np.sin(v)
        y = R * np.sin(u) * np.sin(v)
        z = R * np.cos(v)
        ax.plot_wireframe(x, y, z, color="r")

        
        # colors = ['red', 'green', 'blue', 'yellow']
        # labels = ['Mothership', 'Sat1', 'Sat2', 'Sat3']
        step = 10
        for i in range(num_sat):
            ax.plot(states[::step, i, 0], 
                    states[::step, i, 1], 
                    states[::step, i, 2], 
                    # color=colors[i], 
                    # label=labels[i], 
                    linewidth=1.5)
        
        # Punkty startowe
        ax.scatter(states[0, :, 0], states[0, :, 1], states[0, :, 2], 
                c='black', s=50, marker='o', depthshade=False, label='Start')
        
        # Konfiguracja wykresu
        ax.set_xlabel('X [km]', fontsize=10)
        ax.set_ylabel('Y [km]', fontsize=10)
        ax.set_zlabel('Z [km]', fontsize=10)
        ax.set_title(f'Simulation of {orbit} (RK4)', fontweight='bold')
        
        # Ustawienie jednakowej skali osi
        max_range = np.max(np.abs(states[:, :, :3])) * 1.2
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        # Legenda i siatka
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.5)
        

        if num_sat == 4:
            #Lines between sats
            line_step = 500
            for t_idx in range(0, len(times), line_step):
                positions = states[t_idx]  # shape (4, 3)
                for i, j in combinations(range(4), 2):
                    x = [positions[i, 0], positions[j, 0]]
                    y = [positions[i, 1], positions[j, 1]]
                    z = [positions[i, 2], positions[j, 2]]
                    ax.plot(x, y, z, color='black', linewidth=0.8, alpha=0.5)


        plt.tight_layout()
        plt.show()

    def plot_orbit_3d_plotly(self, times, states, draw_formation = True):
        T = states.shape[0]
        num_sat = states.shape[1]

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

        for i in range(num_sat):
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

        if draw_formation and num_sat == 4:
            line_step = 1000
            for t_idx in range(0, len(times), line_step):
                positions = states[t_idx]
                for i, j in combinations(range(num_sat), 2):
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
        T = states.shape[0]
        num_sat = states.shape[1]

        if num_sat > 4:
            raise Exception("Too many satellites!")
        elif num_sat < 4:
            raise Exception("Not enough satellites")


        if lvlh:
            states = LVLH(states)

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
 
    def plot_errors(self, times, states, sat_separation):
        positions = states[:, :, :3]  # shape (n_steps, 4, 3)
        T = states.shape[0]
        num_sat = states.shape[1]
        
        if num_sat > 4:
            raise Exception("Too many satellites!")
        elif num_sat < 4:
            raise Exception("Not enough satellites")

        dist_errors = np.zeros((T, num_sat))  # 4 satellite pairs
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