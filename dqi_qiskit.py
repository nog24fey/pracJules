import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import RYGate
import matplotlib.pyplot as plt
import networkx as nx
from functools import reduce
import itertools
from scipy.linalg import eigh

# Helper function to get integer from binary array
def _to_int(binary_array):
    # Qiskit uses little-endian, so we don't reverse here.
    return int("".join(str(int(bit)) for bit in binary_array), 2)

# --- DQI Building Blocks ---

# NOTE: The Dicke state preparation from a superposition of unary states is highly non-trivial in Qiskit.
# The Classiq implementation relies on high-level synthesis that is not directly available.
# A full implementation would likely require advanced techniques.
# For this demonstration, we proceed with the superposition of unary states, which have the correct
# Hamming weights but not the full Dicke state superposition. The core logic of DQI still applies.
def prepare_dicke_state_from_unary(qc, max_k, q):
    """
    (Placeholder) Prepares a superposition of Dicke states from a superposition of unary states.
    """
    pass

def vector_product_phase(qc, v, y_q):
    """
    Applies a phase based on the vector product v*y.
    """
    for i in range(len(v)):
        if v[i] > 0:
            qc.z(y_q[i])

def matrix_vector_product(qc, B_T, y_q, out_q):
    """
    Computes the matrix-vector product B^T * y.
    """
    n, m = B_T.shape
    for i in range(n):
        for j in range(m):
            if B_T[i, j] == 1:
                qc.cx(y_q[j], out_q[i])

def syndrome_decode_lookuptable(qc, syndrome_q, error_q, syndromes, errors_map):
    """
    Decodes the syndrome using a lookup table.
    Performs |syndrome>|error> -> |syndrome>|error XOR decoded_error(syndrome)>
    """
    num_syndrome_qubits = len(syndrome_q)

    for s in syndromes:
        e = errors_map[tuple(s)]

        # Calculate ctrl_state integer from little-endian syndrome vector s.
        # s[i] corresponds to syndrome_q[i]. The integer for ctrl_state should be
        # sum(s[i] * 2^i), which is int(little_endian_str[::-1], 2).
        s_str_little_endian = "".join(map(str, s.astype(int)))
        s_int = int(s_str_little_endian[::-1], 2)

        # Create a gate for the error correction
        correction_gate = QuantumCircuit(len(error_q), name=f"corr_{s_int}")
        for bit_idx, bit_val in enumerate(e):
            if bit_val == 1:
                correction_gate.x(bit_idx)

        if not correction_gate.data: # Skip if correction is identity
            continue

        # Apply controlled correction
        controlled_correction = correction_gate.to_gate().control(num_syndrome_qubits, ctrl_state=s_int)
        qc.append(controlled_correction, [*syndrome_q, *error_q])


# --- Full DQI Algorithm ---

def dqi_max_xor_sat(B, v, w_k, syndrome_decode_func, syndromes, errors_map):
    """
    Assembles the full DQI circuit for max-XORSAT.
    """
    m, n = B.shape
    B_T = B.T
    max_errors = len(w_k) - 1 if any(w_k) else 0

    # 1. Prepare registers
    num_k_qubits = (max_errors).bit_length() if max_errors > 0 else 1
    k_q = QuantumRegister(num_k_qubits, 'k')
    y_q = QuantumRegister(m, 'y')
    solution_q = QuantumRegister(n, 'solution')

    # Ancilla for binary_to_unary conversion
    one_hot_size = 2**num_k_qubits
    one_hot_q = QuantumRegister(one_hot_size, 'one_hot_ancilla')

    c_sol = ClassicalRegister(n, 'c_sol')
    c_y = ClassicalRegister(m, 'c_y_debug') # For debugging y

    qc = QuantumCircuit(k_q, y_q, solution_q, one_hot_q, c_sol, c_y)

    # 2. Prepare superposition of |k> states
    w_k_padded = np.zeros(2**num_k_qubits)
    w_k_padded[:len(w_k)] = w_k
    qc.initialize(w_k_padded, k_q)
    qc.barrier(label="k_prepared")

    # 3. Convert |k> to |k_unary> superposition on y_q
    # This version is not in-place but functionally equivalent for the algorithm flow.
    # It maps |k>|0>|0> -> |k>|k_unary>|k_one_hot>
    # We will not uncompute k or one_hot, but treat them as ancillas that are discarded.

    # 3a. Binary to One-Hot (controlled by k_q, output to one_hot_q)
    for k_val in range(one_hot_size):
        ctrl_state_str = format(k_val, f'0{num_k_qubits}b')

        # Find qubits that need to be flipped for 0-control
        zero_controls = []
        # Iterate over the big-endian control string
        for i, bit in enumerate(ctrl_state_str):
            if bit == '0':
                # Map big-endian string index to little-endian qubit index
                qubit_index = num_k_qubits - 1 - i
                zero_controls.append(k_q[qubit_index])

        # Flip 0-controls to 1
        if zero_controls:
            qc.x(zero_controls)

        # All controls are now for state 1
        qc.mcx(k_q, one_hot_q[k_val])

        # Flip 0-controls back
        if zero_controls:
            qc.x(zero_controls)

    qc.barrier(label="one_hot")

    # 3b. One-Hot to Unary (controlled by one_hot_q, output to y_q)
    # y_q[j] = 1 if the number k > j.
    for j in range(m):
        for k_val in range(j + 1, one_hot_size):
            qc.cx(one_hot_q[k_val], y_q[j])
    qc.barrier(label="unary")

    # 4. (Placeholder) Prepare Dicke States from Unary states
    prepare_dicke_state_from_unary(qc, max_errors, y_q)

    # 5. Apply the phase
    vector_product_phase(qc, v, y_q)
    qc.barrier(label="phase")

    # 6. Compute |B^T*y> to a new register
    matrix_vector_product(qc, B_T, y_q, solution_q)
    qc.barrier(label="matrix_prod")

    # 7. Uncompute |y> by decoding the syndrome
    syndrome_decode_func(qc, solution_q, y_q, syndromes, errors_map)
    qc.barrier(label="decode")

    # 8. Transform from Hadamard space to function space
    qc.h(solution_q)
    qc.barrier(label="final_H")

    # 9. Measure
    qc.measure(solution_q, c_sol)
    qc.measure(y_q, c_y)

    return qc

if __name__ == '__main__':
    # --- 1. Problem Definition (Max-Cut on a 2-regular graph) ---
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5])
    G.add_edges_from([(3, 4), (3, 2), (4, 1), (1, 5), (5, 0), (2, 0)])

    # B is the incidence matrix, v is all ones for Max-Cut
    B = nx.incidence_matrix(G, oriented=False).toarray().astype(int).T
    v = np.ones(B.shape[0])
    m, n = B.shape

    print("--- Problem Setup ---")
    print(f"Graph: {G.edges}")
    print(f"B matrix (m={m}, n={n}):\n{B}")
    print(f"v vector:\n{v}")

    # --- 2. Decoder Configuration ---
    MAX_ERRORS = 2

    # Generate all error vectors 'y' with Hamming weight <= MAX_ERRORS
    errors = np.array(
        [
            np.array([1 if i in p else 0 for i in range(m)])
            for k in range(MAX_ERRORS + 1)
            for p in itertools.combinations(range(m), k)
        ],
        dtype=int
    )

    # Calculate corresponding syndromes s = B^T @ y
    syndromes = (errors @ B % 2)

    unique_syndromes, unique_indices = np.unique(syndromes, axis=0, return_index=True)

    if len(unique_syndromes) != len(syndromes):
        print("\nWarning: Syndromes are not unique. The code is not perfectly decodable for this l.")
        errors_map = {tuple(s): errors[i] for s, i in zip(unique_syndromes, unique_indices)}
    else:
        errors_map = {tuple(s): e for s, e in zip(syndromes, errors)}

    print(f"\n--- Decoder Setup (l={MAX_ERRORS}) ---")
    print(f"Number of possible errors (|y|<=l): {len(errors)}")
    print(f"Number of unique syndromes: {len(unique_syndromes)}")

    # --- 3. Optimal w_k Coefficients ---
    def get_optimal_w(m_val, n_val, l_val):
        diag = np.zeros(l_val + 1)
        off_diag = [np.sqrt(i * (m_val - i + 1)) for i in range(1, l_val + 1)]
        A = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
        eigenvalues, eigenvectors = eigh(A)
        principal_vector = eigenvectors[:, np.argmax(eigenvalues)]
        return principal_vector / np.linalg.norm(principal_vector)

    w_k_optimal = get_optimal_w(m, n, MAX_ERRORS)
    print(f"\nOptimal w_k vector: {w_k_optimal}")

    # --- 4. Function to run a single DQI test ---
    def run_dqi_test(w_k_vector):
        qc = dqi_max_xor_sat(B, v, w_k_vector, syndrome_decode_lookuptable, unique_syndromes, errors_map)

        backend = Aer.get_backend('qasm_simulator')
        shots = 4096 # Reduced shots for faster testing

        transpiled_qc = transpile(qc, backend)

        job = backend.run(transpiled_qc, shots=shots, memory=True)
        result = job.result()
        memory = result.get_memory()

        parsed_memory = [mem.split() for mem in memory]
        solutions = np.array([[int(c) for c in sol_str[::-1]] for _, sol_str in parsed_memory])
        f_sampled = ((-1) ** (B @ solutions.T + v[:, np.newaxis])).sum(axis=0)

        y_debug_str = [y_str for y_str, _ in parsed_memory]
        num_failed_decodes = sum(1 for s in y_debug_str if '1' in s)

        print(f"  Decoder check: {num_failed_decodes} of {shots} shots had non-zero y.")
        return np.mean(f_sampled)

    # --- 5. Run tests ---
    print("\n--- Running DQI with Optimal w_k ---")
    avg_f_optimal = run_dqi_test(w_k_optimal)
    print(f"  <f_DQI_optimal>: {avg_f_optimal:.4f}")

    print("\n--- Running DQI with 5 Random w_k vectors ---")
    random_results = []
    for i in range(5):
        print(f"Test Run {i+1}/5:")
        w_k_random = np.random.rand(MAX_ERRORS + 1)
        w_k_random /= np.linalg.norm(w_k_random)
        avg_f_random = run_dqi_test(w_k_random)
        random_results.append(avg_f_random)
        print(f"  w_k_random: {w_k_random}")
        print(f"  <f_DQI_random_{i+1}>: {avg_f_random:.4f}")

    # --- 6. Comparison with uniform sampling ---
    all_inputs = np.array(list(itertools.product([0, 1], repeat=n)), dtype=int)
    f_uniform = ((-1) ** (B @ all_inputs.T + v[:, np.newaxis])).sum(axis=0)
    print(f"\n\n--- Final Comparison ---")
    print(f"<f_uniform>: {np.mean(f_uniform):.4f}")
    print(f"<f_DQI_optimal>: {avg_f_optimal:.4f}")
    for i, res in enumerate(random_results):
        print(f"<f_DQI_random_{i+1}>: {res:.4f}")