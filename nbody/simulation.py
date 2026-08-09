import numpy as np
from tqdm import tqdm


def propagate_orbit(state0, t0, tf, h, f, integrator):
    times = np.arange(t0, tf + h, h)
    T = len(times)
    N = state0.shape[0]

    states = np.zeros((T, N, 6))
    states[0] = state0

    for i in tqdm(range(1, T)):
        states[i] = integrator(state=states[i - 1], t=times[i - 1], h=h, f=f)

        if np.any(np.isnan(states[i])):
            print(f"NaN pojawił się w kroku {i}, t = {times[i]}")
            break

    return times, states


def run_models(state0, t0, tf, h, models, integrator):
    states_by_model = {}

    for model_name, model_func in models.items():
        times, states = propagate_orbit(state0, t0, tf, h, model_func, integrator)
        states_by_model[model_name] = states

    return times, states_by_model


def run_integrators(state0, t0, tf, h, model_func, integrators):
    states_by_method = {}

    for method_name, integrator in integrators.items():
        times, states = propagate_orbit(state0, t0, tf, h, model_func, integrator)
        states_by_method[method_name] = states

    return times, states_by_method


def run_steps(state0, t0, tf, steps, model_func, integrator):
    states_by_step = {}

    for h in steps:
        times, states = propagate_orbit(state0, t0, tf, h, model_func, integrator)
        states_by_step[f"h={h / 3600:.0f}h"] = (times, states)

    return states_by_step