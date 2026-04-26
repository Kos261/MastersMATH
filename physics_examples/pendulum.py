import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000

class PendulumSimulator:
    def __init__(self, length=1.0, gravity=9.81, damping=0.0):
        self.L = length
        self.g = gravity
        self.damping = damping
        
        self.dt = 0.01
        self.t = 0
        
        self.state = np.array([0.0, 0.0])
        
        self.theta_history = []
        self.omega_history = []
        
    def pendulum_ode(self, state):
        theta, omega = state
        alpha = -(self.g / self.L) * np.sin(theta) - self.damping * omega
        return np.array([omega, alpha])
    
    def runge_kutta_step(self):
        k1 = self.pendulum_ode(self.state)
        k2 = self.pendulum_ode(self.state + 0.5 * self.dt * k1)
        k3 = self.pendulum_ode(self.state + 0.5 * self.dt * k2)
        k4 = self.pendulum_ode(self.state + self.dt * k3)
        
        self.state += (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.t += self.dt
        
        self.theta_history.append(self.state[0])
        self.omega_history.append(self.state[1])
    
    def set_initial_conditions(self, theta0, omega0):
        self.state = np.array([theta0, omega0])
        self.t = 0
        self.theta_history = [theta0]
        self.omega_history = [omega0]
    
    def simulate(self, duration):
        steps = int(duration / self.dt)
        for _ in range(steps):
            self.runge_kutta_step()
    
    def plot_phase_portrait(self, theta0, omega0, duration):
        self.set_initial_conditions(theta0, omega0)
        self.simulate(duration)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        step = 50
        ax.plot(self.theta_history[::50,:], self.omega_history, 'b-', linewidth=0.5, alpha=0.8)
        ax.plot(theta0, omega0, 'ro', markersize=8, label=f'Start: θ={theta0:.2f}, ω={omega0:.2f}')
        ax.plot(self.theta_history[-1], self.omega_history[-1], 'go', markersize=8, label='Koniec')
        
        ax.set_xlabel('Kąt θ [rad]')
        ax.set_ylabel('Prędkość kątowa ω [rad/s]')
        ax.set_title(f'Portret fazowy - długa symulacja ({duration}s)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig

if __name__ == "__main__":
    pendulum = PendulumSimulator(length=1.0, gravity=9.81, damping=0.0)
    
    fig = pendulum.plot_phase_portrait(theta0=2.5, omega0=0, duration=60*60*5)
    plt.show()