"""Local, coupled-noise timestep pilots; does not change production code.

Run with the project's Python, for example:
  .venv-win/Scripts/python.exe -B validation_dt_20261112/validate_dt.py

Each base interval is represented by four finest-grid Gaussian increments.
The dt and dt/2 increments are sums of these increments, so all resolutions
follow the same underlying Brownian path at shared time points.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import necklace_chain as nc
import parameters as p
import time_stepping as ts


class FixedNoise:
    def __init__(self, z):
        self.z = z
        self.used = False

    def standard_normal(self, size):
        if self.used or tuple(size) != self.z.shape:
            raise ValueError("Expected exactly one noise draw of the prescribed shape")
        self.used = True
        return self.z


def read_end_frames(path):
    rows = []
    with path.open(encoding="utf-8") as stream:
        first_header = None
        for line in stream:
            if line.startswith("# step="):
                if first_header is not None:
                    break
                first_header = line.strip()
            elif line.strip() and not line.startswith("#"):
                rows.append([float(v) for v in line.split()[2:]])
    first = np.asarray(rows).reshape(p.M, p.N, 3)
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 1_000_000))
        tail = stream.read().decode("utf-8")
    final = tail[tail.rfind("# step="):].splitlines()
    rows = [[float(v) for v in line.split()[2:]]
            for line in final if line.strip() and not line.startswith("#")]
    last = np.asarray(rows).reshape(p.M, p.N, 3)
    return {"flow_initial": (first_header, first),
            "stretched": (final[0], last)}


def select_chains(frames, count):
    """Purposive stress sample, not an unbiased population sample."""
    selected = []
    i, j = np.triu_indices(p.N, 1)
    for _, r in frames.values():
        rho = np.linalg.norm(nc.connection_vector(r), axis=-1)
        distance = np.linalg.norm(r[:, i] - r[:, j], axis=-1)
        span = np.ptp(r[..., 0], axis=1)
        for k in (rho.max(axis=1).argmax(), distance.min(axis=1).argmin(),
                  span.argmax(), span.argmin()):
            if int(k) not in selected:
                selected.append(int(k))
    for k in [*np.linspace(0, p.M-1, count, dtype=int), *range(p.M)]:
        if int(k) not in selected:
            selected.append(int(k))
    return selected[:count]


def observables(r):
    centered = r - r.mean(axis=1, keepdims=True)
    return {"mean_span_A": float(np.ptp(r[..., 0], axis=1).mean()),
            "mean_Rg2_A2": float(np.sum(centered**2, axis=-1).mean()),
            "max_rho": float(np.linalg.norm(nc.connection_vector(r), axis=-1).max()/p.Q0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261112, 20261113])
    parser.add_argument("--base-dt", type=float, default=5e6)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "pilot_results.json")
    args = parser.parse_args()
    if args.chunks < 1 or not 1 <= args.chains <= p.M or args.base_dt <= 0:
        parser.error("Require chunks >= 1, 1 <= chains <= M, base-dt > 0")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log = ROOT / "results/run_20261110_135851_611240/configuration.log"
    frames = read_end_frames(log)
    ids = select_chains(frames, args.chains)
    tau = p.GAMMA/(4*p.H*np.sin(np.pi/(2*p.N))**2)
    gradient = p.velocity_gradient(p.WI/(tau/2))
    source_hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                     for name in ("time_stepping.py", "necklace_chain.py", "parameters.py")}
    report = {"arguments": {**vars(args), "output": str(args.output)},
              "code_sha256": source_hashes, "source_log": str(log),
              "chain_ids_one_based": [v+1 for v in ids],
              "sample_type": "purposive stress sample, not population inference",
              "duration_s_per_scenario": args.chunks*args.base_dt*1e-12,
              "scenarios": []}
    original_solve = np.linalg.solve
    counter = [0]

    def counted_solve(*a, **kw):
        counter[0] += 1
        return original_solve(*a, **kw)

    np.linalg.solve = counted_solve
    i, j = np.triu_indices(p.N, 1)
    try:
        for frame_name, (header, all_r) in frames.items():
            for seed in args.seeds:
                initial = all_r[ids].copy()
                initial -= initial.mean(axis=1, keepdims=True)
                states = {factor: initial.copy() for factor in (4, 2, 1)}
                initial_metrics = observables(initial)
                initial_min_pair = float(np.linalg.norm(initial[:,i]-initial[:,j], axis=-1).min()/p.SIGMA)
                records = {factor: {"dt_ps": args.base_dt*factor/4, "steps": 0,
                                    "newton_solves": 0, "max_newton_solves_per_step": 0,
                                    "wall_s": 0., "failure": None, "sampled_max_rho": initial_metrics["max_rho"],
                                    "sampled_min_pair_sigma": initial_min_pair}
                           for factor in states}
                rng = np.random.default_rng(seed)
                scenario = {"frame": frame_name, "source_header": header,
                            "seed": seed, "initial": observables(initial),
                            "candidates": records, "comparisons": []}
                report["scenarios"].append(scenario)
                print(f"START {frame_name} seed={seed} chains={[v+1 for v in ids]}", flush=True)
                for chunk in range(1, args.chunks+1):
                    fine_z = rng.standard_normal((4, *initial.shape))
                    for factor in (4, 2, 1):
                        rec = records[factor]
                        if rec["failure"] is not None:
                            continue
                        dt = rec["dt_ps"]
                        for offset in range(0, 4, factor):
                            z = fine_z[offset:offset+factor].sum(axis=0)/np.sqrt(factor)
                            old = states[factor]
                            count_before = counter[0]
                            begin = perf_counter()
                            options = dict(velocity_gradient=gradient, kBT=p.KBT, gamma=p.GAMMA,
                                           H=p.H, Q0_square=p.Q02, dt=dt,
                                           epsilon_ev=p.EPSILON, sigma_ev=p.SIGMA)
                            try:
                                out = ts.fene_newton_step(r=old, rng=FixedNoise(z), tol=args.tol, **options)
                            except (RuntimeError, np.linalg.LinAlgError, FloatingPointError) as error:
                                rec["wall_s"] += perf_counter()-begin
                                tb = error.__traceback__
                                while tb.tb_next is not None:
                                    tb = tb.tb_next
                                loc = tb.tb_frame.f_locals
                                residual = loc.get("residual")
                                failure = {"chunk": chunk, "offset": offset,
                                           "elapsed_s": ((chunk-1)+offset/4)*args.base_dt*1e-12,
                                           "error": str(error),
                                           "residual_max_A": float(np.max(np.abs(residual))) if residual is not None else None,
                                           "residual_norm_A": float(np.linalg.norm(residual)) if residual is not None else None,
                                           "same_step_tolerance_replays": []}
                                snapshot = args.output.parent / f"failure_{frame_name}_{seed}_{factor}.npz"
                                np.savez_compressed(snapshot, r=old, z=z, dt=dt,
                                                    candidate=loc.get("candidate_position", old))
                                failure["snapshot"] = str(snapshot)
                                for tolerance in (1e-8, 1e-7):
                                    try:
                                        ts.fene_newton_step(r=old, rng=FixedNoise(z), tol=tolerance, **options)
                                        success = True
                                    except (RuntimeError, np.linalg.LinAlgError, FloatingPointError):
                                        success = False
                                    failure["same_step_tolerance_replays"].append({"tol_A": tolerance, "success": success})
                                rec["failure"] = failure
                                print(f"FAIL {frame_name} seed={seed} dt={dt:g} chunk={chunk} residual={failure['residual_max_A']}", flush=True)
                                break
                            rec["wall_s"] += perf_counter()-begin
                            solves = counter[0]-count_before
                            rec["newton_solves"] += solves
                            rec["max_newton_solves_per_step"] = max(rec["max_newton_solves_per_step"], solves)
                            rec["steps"] += 1
                            states[factor] = out
                            if not np.all(np.isfinite(out)):
                                raise FloatingPointError("Returned nonfinite position")
                    if chunk % max(1, args.chunks//20) == 0 or chunk == args.chunks:
                        entry = {"time_s": chunk*args.base_dt*1e-12}
                        for factor in states:
                            if records[factor]["failure"] is not None:
                                continue
                            r = states[factor]
                            metrics = observables(r)
                            entry[str(factor)] = metrics
                            records[factor]["sampled_max_rho"] = max(records[factor]["sampled_max_rho"], metrics["max_rho"])
                            distance = np.linalg.norm(r[:,i]-r[:,j], axis=-1).min()/p.SIGMA
                            records[factor]["sampled_min_pair_sigma"] = min(records[factor]["sampled_min_pair_sigma"], float(distance))
                        if "1" in entry:
                            ref = states[1]
                            for factor in (4,2):
                                if str(factor) not in entry:
                                    continue
                                m, fine = entry[str(factor)], entry["1"]
                                entry[str(factor)]["span_relative_difference_vs_finest"] = (m["mean_span_A"]-fine["mean_span_A"])/max(abs(fine["mean_span_A"]),1e-30)
                                entry[str(factor)]["Rg2_relative_difference_vs_finest"] = (m["mean_Rg2_A2"]-fine["mean_Rg2_A2"])/fine["mean_Rg2_A2"]
                                entry[str(factor)]["position_rms_difference_over_Q0"] = float(np.sqrt(np.mean(np.sum((states[factor]-ref)**2,axis=-1)))/p.Q0)
                        scenario["comparisons"].append(entry)
                    if chunk % max(1,args.chunks//4) == 0:
                        print(f"PROGRESS {frame_name} seed={seed} chunk={chunk}/{args.chunks}",flush=True)
                        args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
                    if all(v["failure"] is not None for v in records.values()):
                        break
                for factor in states:
                    records[factor]["final"] = observables(states[factor])
                args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
                print(f"DONE {frame_name} seed={seed}",flush=True)
    finally:
        np.linalg.solve = original_solve
    report["source_unchanged"] = all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value
                                      for name,value in source_hashes.items())
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    print(f"RESULT {args.output}",flush=True)


if __name__ == "__main__":
    main()
