import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate parity plots comparing DFT labels against a trained MACE model."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="output/sample_config.xyz/training_data.extxyz",
        help="Reference extxyz dataset with DFT energies and forces.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="checkpoints/maceft_sample75C_run-123.model",
        help="Path to trained MACE .model file.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for MACE inference.",
    )
    parser.add_argument(
        "--outdir",
        default="mace_parity_output",
        help="Output directory for parity figures and summary files.",
    )
    return parser.parse_args()


def make_calculator(model_path, device):
    return MACECalculator(
        model_paths=[str(model_path)],
        device=device,
        default_dtype="float64",
    )


def rmse(y_true, y_pred):
    diff = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sqrt(np.mean(diff ** 2)))


def mae(y_true, y_pred):
    diff = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.mean(np.abs(diff)))


def save_csv(path, header, rows):
    with open(path, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")


def parity_limits(ref_values, pred_values):
    vmin = min(float(np.min(ref_values)), float(np.min(pred_values)))
    vmax = max(float(np.max(ref_values)), float(np.max(pred_values)))
    pad = 0.05 * (vmax - vmin) if vmax > vmin else 1.0
    return vmin - pad, vmax + pad


def main():
    args = parse_args()

    dataset_path = Path(args.dataset)
    model_path = Path(args.model)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    frames = read(dataset_path, index=":")
    calculator = make_calculator(model_path, args.device)

    ref_energy_per_atom = []
    pred_energy_per_atom = []
    ref_force_components = []
    pred_force_components = []
    energy_rows = []
    force_rows = []

    print("=" * 72)
    print("Running MACE parity evaluation")
    print(f"Dataset: {dataset_path}")
    print(f"Model:   {model_path}")
    print(f"Device:  {args.device}")
    print(f"Frames:  {len(frames)}")
    print("=" * 72)

    for i, atoms in enumerate(frames):
        ref_energy = atoms.get_potential_energy()
        ref_forces = atoms.get_forces()

        pred_atoms = atoms.copy()
        pred_atoms.calc = calculator

        pred_energy = pred_atoms.get_potential_energy()
        pred_forces = pred_atoms.get_forces()

        natoms = len(atoms)
        ref_e_pa = ref_energy / natoms
        pred_e_pa = pred_energy / natoms

        ref_energy_per_atom.append(ref_e_pa)
        pred_energy_per_atom.append(pred_e_pa)
        energy_rows.append((i, natoms, ref_e_pa, pred_e_pa, pred_e_pa - ref_e_pa))

        flat_ref_forces = ref_forces.reshape(-1)
        flat_pred_forces = pred_forces.reshape(-1)
        ref_force_components.extend(flat_ref_forces.tolist())
        pred_force_components.extend(flat_pred_forces.tolist())

        for j, (f_ref, f_pred) in enumerate(zip(flat_ref_forces, flat_pred_forces)):
            force_rows.append((i, j, f_ref, f_pred, f_pred - f_ref))

        if (i + 1) % 25 == 0 or i == len(frames) - 1:
            print(f"Processed {i + 1}/{len(frames)} frames")

    ref_energy_per_atom = np.array(ref_energy_per_atom)
    pred_energy_per_atom = np.array(pred_energy_per_atom)
    ref_force_components = np.array(ref_force_components)
    pred_force_components = np.array(pred_force_components)

    energy_rmse = rmse(ref_energy_per_atom, pred_energy_per_atom)
    energy_mae = mae(ref_energy_per_atom, pred_energy_per_atom)
    force_rmse = rmse(ref_force_components, pred_force_components)
    force_mae = mae(ref_force_components, pred_force_components)

    energy_csv = outdir / "energy_parity.csv"
    force_csv = outdir / "force_parity.csv"
    save_csv(
        energy_csv,
        "frame,natoms,ref_energy_per_atom_eV,pred_energy_per_atom_eV,error_eV_per_atom",
        energy_rows,
    )
    save_csv(
        force_csv,
        "frame,component,ref_force_eVA,pred_force_eVA,error_eVA",
        force_rows,
    )

    e_min, e_max = parity_limits(ref_energy_per_atom, pred_energy_per_atom)
    f_min, f_max = parity_limits(ref_force_components, pred_force_components)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

    axes[0].scatter(ref_energy_per_atom, pred_energy_per_atom, s=18, alpha=0.75)
    axes[0].plot([e_min, e_max], [e_min, e_max], "k--", lw=1)
    axes[0].set_xlim(e_min, e_max)
    axes[0].set_ylim(e_min, e_max)
    axes[0].set_xlabel("DFT energy / atom (eV)")
    axes[0].set_ylabel("MACE energy / atom (eV)")
    axes[0].set_title("Energy Parity")
    axes[0].grid(True, alpha=0.3)
    axes[0].text(
        0.04,
        0.96,
        f"MAE = {energy_mae * 1000:.2f} meV/atom\nRMSE = {energy_rmse * 1000:.2f} meV/atom",
        transform=axes[0].transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    axes[1].scatter(ref_force_components, pred_force_components, s=6, alpha=0.25)
    axes[1].plot([f_min, f_max], [f_min, f_max], "k--", lw=1)
    axes[1].set_xlim(f_min, f_max)
    axes[1].set_ylim(f_min, f_max)
    axes[1].set_xlabel("DFT force component (eV/A)")
    axes[1].set_ylabel("MACE force component (eV/A)")
    axes[1].set_title("Force Parity")
    axes[1].grid(True, alpha=0.3)
    axes[1].text(
        0.04,
        0.96,
        f"MAE = {force_mae * 1000:.1f} meV/A\nRMSE = {force_rmse * 1000:.1f} meV/A",
        transform=axes[1].transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    fig.tight_layout()
    parity_png = outdir / "mace_parity.png"
    fig.savefig(parity_png, dpi=600, bbox_inches="tight")
    plt.close(fig)

    summary_path = outdir / "mace_parity_summary.txt"
    with open(summary_path, "w") as f:
        f.write("MACE Parity Summary\n")
        f.write("===================\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Frames: {len(frames)}\n")
        f.write(f"Force components: {len(ref_force_components)}\n")
        f.write(f"Energy MAE  = {energy_mae:.10f} eV/atom\n")
        f.write(f"Energy RMSE = {energy_rmse:.10f} eV/atom\n")
        f.write(f"Force MAE   = {force_mae:.10f} eV/A\n")
        f.write(f"Force RMSE  = {force_rmse:.10f} eV/A\n")

    print("\n" + "=" * 72)
    print("Parity evaluation complete")
    print(f"Energy MAE  = {energy_mae * 1000:.2f} meV/atom")
    print(f"Energy RMSE = {energy_rmse * 1000:.2f} meV/atom")
    print(f"Force MAE   = {force_mae * 1000:.2f} meV/A")
    print(f"Force RMSE  = {force_rmse * 1000:.2f} meV/A")
    print(f"Saved plot to:    {parity_png}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved CSVs to:    {energy_csv} and {force_csv}")
    print("=" * 72)


if __name__ == "__main__":
    main()
