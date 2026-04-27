import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ase.io import read
from ase.eos import EquationOfState
from ase.units import kJ

from gpaw import GPAW, PW


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute DFT EOS and bulk modulus using GPAW."
    )
    parser.add_argument(
        "-c", "--config",
        default="output/sample_config.xyz/relax.xyz",
        help="Input structure file (recommended: relaxed structure)."
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
        "--min-scale",
        type=float,
        default=0.94,
        help="Minimum isotropic cell scale factor."
    )
    parser.add_argument(
        "--max-scale",
        type=float,
        default=1.06,
        help="Maximum isotropic cell scale factor."
    )
    parser.add_argument(
        "--npoints",
        type=int,
        default=11,
        help="Number of scale points for EOS."
    )
    parser.add_argument(
        "--outdir",
        default="gpaw_eos_output",
        help="Output directory."
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


def main():
    args = parse_args()

    config_path = Path(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    atoms0 = read(config_path)
    original_cell = atoms0.get_cell()

    scales = np.linspace(args.min_scale, args.max_scale, args.npoints)

    volumes = []
    energies = []

    print("=" * 72)
    print("Running GPAW EOS calculation")
    print(f"Structure: {config_path}")
    print(f"ecut:      {args.ecut} eV")
    print(f"kpts:      {tuple(args.kpts)}")
    print(f"Scales:    {args.min_scale:.4f} -> {args.max_scale:.4f} ({args.npoints} points)")
    print("=" * 72)

    for i, scale in enumerate(scales):
        atoms = atoms0.copy()
        atoms.set_cell(original_cell * scale, scale_atoms=True)

        log_path = outdir / f"gpaw_eos_{i:02d}_scale_{scale:.4f}.txt"
        atoms.calc = make_calc(log_path, args.ecut, args.kpts)

        energy = atoms.get_potential_energy()
        volume = atoms.get_volume()

        volumes.append(volume)
        energies.append(energy)

        print(
            f"[{i+1:02d}/{len(scales):02d}] "
            f"scale={scale:.4f}  volume={volume:.6f} A^3  energy={energy:.8f} eV"
        )

    volumes = np.array(volumes)
    energies = np.array(energies)

    eos = EquationOfState(volumes, energies)
    v0, e0, B = eos.fit()
    B_GPa = B / kJ * 1.0e24

    # Save raw data
    csv_path = outdir / "gpaw_eos_results.csv"
    with open(csv_path, "w") as f:
        f.write("scale,volume_A3,energy_eV\n")
        for s, v, e in zip(scales, volumes, energies):
            f.write(f"{s:.8f},{v:.12f},{e:.12f}\n")

    # ASE EOS plot
    png_path = outdir / "gpaw_eos.png"
    eos.plot(str(png_path))

    # Simple custom plot
    custom_png = outdir / "gpaw_eos_custom.png"
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    ax.plot(volumes, energies, "o", label="GPAW points")
    ax.set_xlabel(r"Volume ($\AA^3$)")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("GPAW EOS")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(custom_png, dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Save summary
    summary_path = outdir / "gpaw_eos_summary.txt"
    with open(summary_path, "w") as f:
        f.write("GPAW EOS Summary\n")
        f.write("================\n")
        f.write(f"Structure: {config_path}\n")
        f.write(f"ecut: {args.ecut} eV\n")
        f.write(f"kpts: {tuple(args.kpts)}\n")
        f.write(f"V0 = {v0:.10f} A^3\n")
        f.write(f"E0 = {e0:.10f} eV\n")
        f.write(f"B  = {B_GPa:.6f} GPa\n")

    print("\n" + "=" * 72)
    print("EOS fit complete")
    print(f"Equilibrium volume V0 = {v0:.6f} A^3")
    print(f"Minimum energy     E0 = {e0:.8f} eV")
    print(f"Bulk modulus        B = {B_GPa:.4f} GPa")
    print(f"Saved raw data to:      {csv_path}")
    print(f"Saved ASE EOS plot to:  {png_path}")
    print(f"Saved custom plot to:   {custom_png}")
    print(f"Saved summary to:       {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
