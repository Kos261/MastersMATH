from hmac import digest

import numpy as np
from tqdm import tqdm
from pathlib import Path
import hashlib
import json

def make_cache_name(integrator_name, model_name, bodies, t0_str, tf_str, h, version="v1"):
    config = {
        "integrator_name": integrator_name,
        "model_name": model_name,
        "bodies": bodies,
        "t0_str": t0_str,
        "tf_str": tf_str,
        "h": h,
        "version": version
    }
    config_string = json.dumps(config, sort_keys=True)
    digest = hashlib.sha1(config_string.encode()).hexdigest()[0:4]
    h_hours = h / 3600
    h_tag = f"{h_hours:g}h".replace(".", "p")
    return f"{integrator_name}__{model_name}__n{len(bodies)}__{t0_str[:10]}__{tf_str[:10]}__h{h_tag}__{version}__{digest}.npz"


def propagate_cached(state0, t0, tf, h, f, integrator, cache_dir, cache_name):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    path = cache_dir / cache_name

    if path.exists():
        print(f"Loading: {path}")
        data = np.load(path)
        return data["times"], data["states"]

    times, states = propagate_orbit(state0, t0, tf, h, f, integrator)

    print(f"Saving: {path}")
    np.savez_compressed(path, times=times, states=states)

    return times, states


def propagate_orbit(state0, t0, tf, h, f, integrator):
    times = np.arange(t0, tf + h, h)
    T = len(times)
    N = state0.shape[0]

    states = np.zeros((T, N, 6))
    states[0] = state0

    # for i in tqdm(range(1, T)):
    for i in range(1,T):
        states[i] = integrator(
            state=states[i - 1],
            t=times[i - 1],
            h=h,
            f=f
        )

    return times, states


def run_models(state0, t0, tf, h, models, integrator):
    states_by_model = {}

    for model_name, model_func in models.items():
        times, states = propagate_orbit(state0, t0, tf, h, model_func, integrator)
        states_by_model[model_name] = states

    return times, states_by_model


def run_integrators(state0, t0, tf, h, model_func, integrators, filename=None):
    states_by_method = {}

    for method_name, integrator in integrators.items():
        times, states = propagate_orbit(state0, t0, tf, h, model_func, integrator, filename)
        states_by_method[method_name] = states

    return times, states_by_method


def run_steps(state0, t0, tf, steps, model_func, integrator):
    states_by_step = {}

    for h in steps:
        times, states = propagate_orbit(state0, t0, tf, h, model_func, integrator)
        states_by_step[f"h={h / 3600:.0f}h"] = (times, states)

    return states_by_step



def energy_metrics(times, H):
    time_days = (times - times[0]) / 86400.0
    e = (H - H[0]) / abs(H[0])

    slope, intercept = np.polyfit(time_days, e, 1)
    trend = intercept + slope * time_days
    residual = e - trend

    T = time_days[-1] - time_days[0]
    drift = abs(slope) * T
    oscillation = np.sqrt(np.mean(residual**2))
    ratio = drift / oscillation if oscillation > 0 else np.inf
    crossover_days = oscillation / abs(slope) if abs(slope) > 0 else np.inf

    return {
        "slope": slope,
        "drift": drift,
        "oscillation": oscillation,
        "ratio": ratio,
        "crossover_days": crossover_days,
    }