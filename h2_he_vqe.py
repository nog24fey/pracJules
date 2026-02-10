import numpy as np
import matplotlib.pyplot as plt
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import StatevectorEstimator

def run_h2_vqe():
    print("Starting H2 Dissociation Curve Calculation using HE-VQE (Hardware Efficient)...")
    print("Note: This implements a standard Hardware Efficient VQE (EfficientSU2).")
    print("      'HI-VQE' (Handover Iterative) is a specific proprietary/recent algorithm")
    print("      not available in standard libraries, so we use the standard HE-VQE.")

    distances = np.arange(0.5, 2.6, 0.1)
    vqe_energies = []
    exact_energies = []

    # Mapper
    mapper = JordanWignerMapper()

    # Optimizer
    optimizer = COBYLA(maxiter=1000)

    # Estimator (Exact simulation)
    # Use StatevectorEstimator (V2 primitive) as recommended for Qiskit 1.x+
    estimator = StatevectorEstimator()

    for dist in distances:
        print(f"Processing distance: {dist:.2f} Angstrom...")

        # 1. Define Molecule
        driver = PySCFDriver(
            atom=f"H 0 0 0; H 0 0 {dist}",
            charge=0,
            spin=0,
            basis='sto-3g',
            unit=DistanceUnit.ANGSTROM
        )
        problem = driver.run()

        # 2. Exact Solver (for reference)
        solver_exact = GroundStateEigensolver(mapper, NumPyMinimumEigensolver())
        result_exact = solver_exact.solve(problem)
        exact_e = result_exact.total_energies[0]
        exact_energies.append(exact_e)

        # 3. VQE Solver with Hardware Efficient Ansatz
        # H2 STO-3G -> 4 Spin Orbitals -> 4 Qubits with Jordan-Wigner
        # EfficientSU2 is a standard "Hardware Efficient" ansatz
        ansatz = EfficientSU2(num_qubits=4, reps=2, entanglement='linear')

        vqe = VQE(estimator, ansatz, optimizer)
        solver_vqe = GroundStateEigensolver(mapper, vqe)

        result_vqe = solver_vqe.solve(problem)
        vqe_e = result_vqe.total_energies[0]
        vqe_energies.append(vqe_e)

        print(f"  Result -> Exact: {exact_e:.5f} Ha, VQE: {vqe_e:.5f} Ha")

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(distances, exact_energies, 'k-', label='Exact (FCI)')
    plt.plot(distances, vqe_energies, 'r--o', label='HE-VQE (EfficientSU2)')
    plt.xlabel('Interatomic Distance (Angstrom)')
    plt.ylabel('Energy (Hartree)')
    plt.title('H2 Dissociation Curve: HE-VQE vs Exact')
    plt.legend()
    plt.grid(True)
    plt.savefig('h2_dissociation_curve.png')
    print("Plot saved to h2_dissociation_curve.png")

if __name__ == "__main__":
    run_h2_vqe()
