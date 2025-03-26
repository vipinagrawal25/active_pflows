"""
Nicholas M. Boffi
4/1/24

Generate a dataset for the MIPS system.
"""
import jax
import numpy as onp
from jax import numpy as np
from jax import vmap, jit
import dill as pickle
import sys

sys.path.append("../")
import common.drifts as drifts
import argparse
import time
from functools import partial
from tqdm.auto import tqdm as tqdm
import yaml  # Add import for YAML

@partial(jit, static_argnums=(6, 7))
def rollout(
    init_xg: np.ndarray,  # [2N, d]
    noises: np.ndarray,  # [nsteps, N, d]
    dt: float,
    radii: np.ndarray,
    A: float,
    k: float,
    v0: float,
    N: int,
    d: int,
    eps: float,
    gamma: float,
    width: float,
    beta: float,
) -> np.ndarray:
    """Rollout the MIPS system for nsteps steps."""
    print("Jitting the rollout...")

    def scan_fn(xg: np.ndarray, noise: np.ndarray):
        xgnext = drifts.step_mips_OU_EM(
            xg, dt, radii, A, k, v0, N, d, eps, gamma, width, beta, noise
        )
        return xgnext, xgnext

    xg_final, xg_traj = jax.lax.scan(scan_fn, init_xg, noises)
    return xg_final

def load_parameters_from_yaml(file_path: str):
    """Load parameters from a YAML file."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    ## standard system parameters
    print("Entering main. Loading parameters from input.yaml.")
    d = 2
    r = 1.0
    n_prints = 100

    ## load parameters from input.yaml
    params = load_parameters_from_yaml("input.yaml")
    gamma = params["gamma"]
    N = params["N"]
    v0 = params["v0"]
    A = params["A"]
    k = params["k"]
    eps = params["eps"]
    beta = params["beta"]
    phi = params["phi"]
    dt = params["dt"]
    ndata = params["ndata"]
    output_folder = params["output_folder"]
    radii = onp.ones(N) * r
    width = np.sqrt(np.sum(radii**2) * np.pi / phi) / 2
    dim = 2 * N * d
    sig0x, sig0g = width / 2, 1.0

    ## thermalization parameters
    nbatches = params["nbatches"]
    nbatches_space = params["nbatches_space"]
    thermalize_fac = params["thermalize_fac"]
    tspace = params["tspace"]
    divide_fac = min(gamma, eps)
    divide_fac = max(gamma, eps) if divide_fac == 0 else divide_fac
    tf = thermalize_fac / divide_fac
    nsteps_thermalize = int((tf / dt) // nbatches) + 1
    nsteps_space = int((tspace / dt) // nbatches_space) + 1
    key = jax.random.PRNGKey(onp.random.randint(10000))

    ## set up the trajectory storage
    traj = onp.zeros((ndata + 1, 2 * N, d))

    # uniform initial conditions for gas
    if phi < 0.25:
        xs = drifts.torus_project(
            width * onp.random.uniform(-1, 1, size=(N * d)), width
        )
    # gaussian for solids
    else:
        xs = drifts.torus_project(sig0x * onp.random.randn(N * d), width)

    gs = sig0g * onp.random.randn(N * d)
    xgs = onp.concatenate((xs, gs)).reshape((2 * N, d))
    name = f"OU_v0={v0}_N={N}_gamma={gamma}_phi={phi}_dt={dt}_beta={beta}_tspace={tspace}_A={A}_k={k}_eps={eps}"

    # log some info
    print(f"Starting dataset generation.")
    print(f"Output: {output_folder}/{name}.npy")

    # set up output data storage
    data_dict = {
        "gamma": gamma,
        "eps": eps,
        "v0": v0,
        "k": k,
        "A": A,
        "beta": beta,
        "phi": phi,
        "dt": dt,
        "tspace": tspace,
        "tf_thermalize": tf,
        "width": width,
        "r": r,
        "d": d,
        "N": N,
    }

    ## thermalize
    if params["load_thermalized"]:
        print("Loading thermalized data.")
        thermalized_data = pickle.load(open(params["thermalized_location"], "rb"))
        xgs = thermalized_data["traj"][-1]
    else:
        for curr_batch in tqdm(range(nbatches)):
            print(f"Starting thermal batch {curr_batch+1}/{nbatches}")
            batch_start = time.time()
            noises = jax.random.normal(key, shape=(nsteps_thermalize, 2 * N, d))
            key = jax.random.split(key)[0]
            xgs = rollout(
                xgs, noises, dt, radii, A, k, v0, N, d, eps, gamma, width, beta
            )
            batch_end = time.time()
            print(
                f"Finished thermal batch {curr_batch+1}/{nbatches}. Time: {(batch_end-batch_start)/60}m."
            )
        print(f"Finished thermalizing.")

    ## temporal dataset
    start_time = time.time()
    traj[0] = xgs
    for curr_datapt in tqdm(range(ndata)):
        for curr_batch in range(nbatches_space):
            noises = jax.random.normal(key, shape=(nsteps_space, 2 * N, d))
            key = jax.random.split(key)[0]
            xgs = rollout(
                xgs, noises, dt, radii, A, k, v0, N, d, eps, gamma, width, beta
            )
        traj[curr_datapt + 1] = xgs

        try:
            if (curr_datapt % int(ndata // n_prints)) == 0:
                end_time = time.time()
                print(f"Finished data point {curr_datapt+1}/{ndata}.")
                print(f"Total time: {(end_time - start_time)/60}m.")
                start_time = time.time()
                data_dict["traj"] = traj
                pickle.dump(data_dict, open(f"{output_folder}/{name}.npy", "wb"))
        except:
            print("Too few data points to print progress.")

    data_dict["traj"] = traj
    pickle.dump(data_dict, open(f"{output_folder}/{name}_{slurm_id}.npy", "wb"))