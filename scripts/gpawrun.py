from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.optimize import FIRE
from ase import units
from gpaw import GPAW, PW
from gpaw.mpi import world

import argparse
from pathlib import Path
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate DFT-MD data and export an extxyz dataset for MLIP training."
    )
    parser.add_argument(
        "-c", "--config",
        default="sample_config.xyz",
        help="Input structure file (.xyz/.extxyz/.data)."
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Default: output/<config filename>"
    )
    parser.add_argument(
        "--ecut",
        type=float,
        default=400.0,
        help="Plane-wave cutoff energy in eV."
    )
    parser.add_argument(
        "--kpts",
        type=int,
        nargs=3,
        default=[1, 1, 1],
        help="Monkhorst-Pack k-point grid, e.g. --kpts 1 1 1"
    )
    parser.add_argument(
        "--relax-steps",
        type=int,
        default=150,
        help="Maximum number of relaxation steps."
    )
    parser.add_argument(
        "--relax-fmax",
        type=float,
        default=0.08,
        help="Force convergence threshold for relaxation in eV/Angstrom."
    )
    parser.add_argument(
        "--md-steps",
        type=int,
        default=400,
        help="Number of MD steps per temperature."
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=5,
        help="Write one trajectory frame every N MD steps."
    )
    parser.add_argument(
        "--time-step-fs",
        type=float,
        default=1.0,
        help="MD timestep in femtoseconds."
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=0.02,
        help="Langevin friction coefficient."
    )
    parser.add_argument(
        "--temperatures",
        type=int,
        nargs="+",
        default=[300, 600, 900],
        help="List of MD temperatures in Kelvin."
    )
    return parser.parse_args()


def make_calc(logfile, ecut, kpts):
    return GPAW(
        mode=PW(ecut),
        xc="PBE",
        symmetry="off",
        kpts=tuple(kpts),
        txt=str(logfile),
    )


def freeze_results(atoms, label, step=None):
    """
    Evaluate and store energy/forces explicitly in extxyz-friendly fields,
    then detach the calculator to avoid duplicate ASE extxyz save conflicts.
    """
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    atoms.info["config_type"] = label
    atoms.info["energy"] = float(energy)
    if step is not None:
        atoms.info["step"] = int(step)

    atoms.arrays["forces"] = forces.copy()

    # Prevent ASE extxyz writer from trying to save calculator results again
    atoms.calc = None
    return atoms


def append_extxyz(dataset_path, frames):
    if not frames:
        return
    write(dataset_path, frames, format="extxyz", append=dataset_path.exists())


def load_structure(config_path):
    if config_path.suffix in {".xyz", ".extxyz"}:
        return read(config_path, index="-1")
    if config_path.suffix == ".data":
        return read(config_path, format="lammps-data")
    raise ValueError(f"Unsupported config format: {config_path.suffix}")


def main():
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.is_file():
        print(f"Config file does not exist: {config_path}")
        sys.exit(1)

    if world.rank == 0:
        print("=" * 72)
        print("DFT-MD dataset generation for MLIP fine-tuning")
        print(f"Running on {world.size} MPI ranks")
        print(f"Input config: {config_path}")
        print(f"ecut = {args.ecut} eV")
        print(f"kpts = {tuple(args.kpts)}")
        print(f"relax_steps = {args.relax_steps}")
        print(f"relax_fmax = {args.relax_fmax} eV/Angstrom")
        print(f"md_steps = {args.md_steps}")
        print(f"sample_every = {args.sample_every}")
        print(f"time_step_fs = {args.time_step_fs}")
        print(f"friction = {args.friction}")
        print(f"temperatures = {args.temperatures}")
        print("=" * 72)

    atoms = load_structure(config_path)

    outdir = Path(args.outdir) if args.outdir else Path("output") / config_path.name
    os.makedirs(outdir, exist_ok=True)

    dataset_path = outdir / "training_data.extxyz"
    if world.rank == 0 and dataset_path.exists():
        dataset_path.unlink()

    # ---------------------------
    # Relaxation
    # ---------------------------
    if world.rank == 0:
        print("Starting structure relaxation...")

    atoms.calc = make_calc(outdir / "gpaw_relax.txt", args.ecut, args.kpts)

    relax = FIRE(atoms)
    relax_traj_path = outdir / "relax.traj"
    relax_traj = Trajectory(relax_traj_path, "w", atoms)
    relax.attach(relax_traj.write, interval=1)
    relax.run(fmax=args.relax_fmax, steps=args.relax_steps)
    relax_traj.close()

    relaxed = atoms.copy()
    relaxed.calc = atoms.calc
    relaxed = freeze_results(relaxed, "relaxed", step=0)

    write(outdir / "relax.xyz", [relaxed], format="extxyz")
    append_extxyz(dataset_path, [relaxed])

    if world.rank == 0:
        print(f"Relaxed structure written to: {outdir / 'relax.xyz'}")

    # ---------------------------
    # DFT-MD sampling
    # ---------------------------
    time_step = args.time_step_fs * units.fs

    for temperature in args.temperatures:
        if world.rank == 0:
            print(f"Starting MD at {temperature} K...")

        md_atoms = relaxed.copy()
        md_atoms.calc = make_calc(
            outdir / f"gpaw_md_{temperature}.txt",
            args.ecut,
            args.kpts,
        )

        dyn = Langevin(
            md_atoms,
            timestep=time_step,
            temperature_K=temperature,
            friction=args.friction,
            logfile=str(outdir / f"{temperature}.log"),
            loginterval=1,
            fixcm=False,
        )

        md_traj_path = outdir / f"{temperature}.traj"
        md_traj = Trajectory(md_traj_path, "w", md_atoms)
        dyn.attach(md_traj.write, interval=args.sample_every)
        dyn.run(args.md_steps)
        md_traj.close()

        # Reload trajectory and relabel each saved frame with explicit results
        frames = read(md_traj_path, index=":")
        labeled_frames = []

        for i, frame in enumerate(frames):
            step_number = i * args.sample_every

            # Reattach a fresh calculator for a clean single-point evaluation
            frame.calc = make_calc(
                outdir / f"gpaw_sp_{temperature}_{i:04d}.txt",
                args.ecut,
                args.kpts,
            )

            frame = freeze_results(frame, f"md_{temperature}K", step=step_number)
            labeled_frames.append(frame)

        xyz_path = outdir / f"{temperature}.xyz"
        write(xyz_path, labeled_frames, format="extxyz")
        append_extxyz(dataset_path, labeled_frames)

        if world.rank == 0:
            print(f"Wrote {len(labeled_frames)} labeled frames to: {xyz_path}")

    if world.rank == 0:
        print("=" * 72)
        print(f"Training dataset written to: {dataset_path}")
        print("Done.")
        print("=" * 72)


if __name__ == "__main__":
    main()