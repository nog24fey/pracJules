# -*- coding: utf-8 -*-
"""
Qiskitを用いた1次元熱方程式の量子シミュレーション

このスクリプトは、1次元の熱伝導方程式を、古典的な差分法と
量子コンピュータ（シミュレータ）を用いた量子アルゴリズムの両方で解き、
その結果を比較・可視化するものです。

アルゴリズムの概要:
1. 古典シミュレーション:
   - 時間方向は前進差分、空間方向は中心差分を用いる陽解法で方程式を解きます。
   - NumPyを用いて、時間ステップごとに温度分布を計算します。

2. 量子シミュレーション:
   - 温度分布を量子状態の振幅としてエンコードします。
   - 熱方程式の空間微分（ラプラシアン）に対応するハミルトニアンを構築します。
   - シュレーディンガー方程式に基づく時間発展演算子 U = exp(-i*H*dt) を計算します。
     - 本実装では、この演算子をScipyで直接計算し、ユニタリゲートとして回路に適用します。
       これにより、シミュレータ上での計算を簡略化しています。
   - QiskitのStatevectorSimulatorを用いて、各時間ステップ後の量子状態を計算します。
   - 状態ベクトルの各振幅の2乗を、各地点の温度に対応するものとして解釈します。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
from scipy.linalg import expm

# --- 量子シミュレーション関連 ---

def get_hamiltonian_matrix(n_qubits, alpha, dx):
    """
    熱方程式の時間発展に対応するハミルトニアン行列を構築します。
    ハミルトニアン H は、空間の2階差分（ラプラシアン）演算子 D を用いて
    H = -alpha * D / dx^2 として定義されます。
    """
    N = 2**n_qubits
    # 中の要素を定義
    D = np.diag(np.ones(N-1), -1) - 2 * np.diag(np.ones(N), 0) + np.diag(np.ones(N-1), 1)
    # 周期的境界条件を適用
    D[0, N-1] = 1
    D[N-1, 0] = 1

    return -alpha * D / (dx**2)

def solve_quantum_heat_equation(n_qubits, n_time_steps, total_time, alpha, L=1.0):
    """
    Qiskitを使用して1次元の熱方程式をシミュレートします。
    """
    nx = 2**n_qubits
    dx = L / (nx - 1) if nx > 1 else L
    dt = total_time / n_time_steps

    x = np.linspace(0, L, nx)

    # 1. 初期状態の準備: sin(pi*x)
    initial_state_vector = np.sin(np.pi * x)
    initial_state_vector /= np.linalg.norm(initial_state_vector)

    # 2. ハミルトニアンの構築
    H = get_hamiltonian_matrix(n_qubits, alpha, dx)

    # 3. 時間発展演算子の計算
    time_evolution_operator = expm(-1j * H * dt)

    # 4. Qiskitによるシミュレーション
    u_quantum = np.zeros((n_time_steps + 1, nx))
    u_quantum[0, :] = np.abs(initial_state_vector)**2

    simulator = Aer.get_backend('statevector_simulator')
    current_state = initial_state_vector

    for t in range(n_time_steps):
        q = QuantumRegister(n_qubits)
        step_qc = QuantumCircuit(q)
        step_qc.initialize(current_state, q)
        step_qc.unitary(time_evolution_operator, q)

        result = simulator.run(step_qc).result()
        current_state = result.get_statevector().data

        u_quantum[t + 1, :] = np.abs(current_state)**2

    return x, u_quantum

# --- 古典シミュレーション関連 ---

def solve_classical_heat_equation(nx, nt, alpha, L=1.0, T=0.1):
    """
    1次元の熱方程式を古典的な差分法（陽解法）で解きます。
    """
    dx = L / (nx - 1) if nx > 1 else L
    dt = T / nt
    r = alpha * dt / dx**2

    if r > 0.5:
        print(f"警告: 計算が不安定になる可能性があります。r = {r:.4f} > 0.5")

    x = np.linspace(0, L, nx)
    u = np.zeros((nt + 1, nx))
    u[0, :] = np.sin(np.pi * x)
    u[:, 0] = 0
    u[:, -1] = 0

    for t in range(nt):
        u[t + 1, 1:-1] = u[t, 1:-1] + r * (u[t, 2:] - 2 * u[t, 1:-1] + u[t, 0:-2])

    return x, u

# --- メイン処理 ---

if __name__ == '__main__':
    # --- シミュレーションパラメータ設定 ---
    N_QUBITS = 4
    NX = 2**N_QUBITS
    ALPHA = 0.001
    L = 1.0
    TOTAL_TIME = 0.2
    NT_CLASSICAL = 2000
    N_TIME_STEPS_QUANTUM = 50
    ANIMATION_FRAMES = 50 # アニメーションのフレーム数

    # --- シミュレーションの実行 ---
    print("Running classical heat equation simulation...")
    x_classical, u_classical = solve_classical_heat_equation(
        nx=NX, nt=NT_CLASSICAL, alpha=ALPHA, L=L, T=TOTAL_TIME
    )
    print("\nRunning quantum heat equation simulation...")
    x_quantum, u_quantum_prob = solve_quantum_heat_equation(
        n_qubits=N_QUBITS, n_time_steps=N_TIME_STEPS_QUANTUM,
        total_time=TOTAL_TIME, alpha=ALPHA, L=L
    )
    u_quantum = u_quantum_prob * np.sum(u_classical[0, :]) / np.sum(u_quantum_prob[0, :])

    # --- アニメーションの生成 ---
    print("\nCreating animation...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)
    fig.suptitle('1D Heat Equation Simulation: Classical vs. Quantum', fontsize=16)

    y_max = np.max(u_classical[0, :]) * 1.1
    ax1.set_ylim(0, y_max)

    # プロットオブジェクトの初期化
    line1, = ax1.plot([], [], 'b-', label='Classical')
    ax1.set_title("Classical Simulation")
    ax1.set_ylabel("Temperature")
    ax1.grid(True)
    ax1.legend()

    line2, = ax2.plot([], [], 'r--', label='Quantum')
    ax2.set_title("Quantum Simulation")
    ax2.set_xlabel("Position")
    ax2.set_ylabel("Temperature (from prob.)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 各フレームの更新処理
    def update(frame):
        # 古典データのインデックスを計算
        classical_idx = int(frame * (NT_CLASSICAL / ANIMATION_FRAMES))
        line1.set_data(x_classical, u_classical[classical_idx, :])

        # 量子データのインデックスを計算
        quantum_idx = int(frame * (N_TIME_STEPS_QUANTUM / ANIMATION_FRAMES))
        if quantum_idx > N_TIME_STEPS_QUANTUM:
            quantum_idx = N_TIME_STEPS_QUANTUM
        line2.set_data(x_quantum, u_quantum[quantum_idx, :])

        return line1, line2

    # アニメーションの作成と保存
    ani = animation.FuncAnimation(fig, update, frames=ANIMATION_FRAMES, blit=True)

    # GIFとして保存
    try:
        ani.save('comparison_heat_animation.gif', writer='imagemagick', fps=10)
        print("\nAnimation saved to 'comparison_heat_animation.gif'")
    except Exception as e:
        print(f"\nError saving animation: {e}")
        print("Please ensure you have 'imagemagick' installed and configured in your environment.")
        print("Saving as a static plot instead.")
        # 静的プロットのフォールバック
        ax1.plot(x_classical, u_classical[-1, :], 'b-')
        ax2.plot(x_quantum, u_quantum[-1, :], 'r--')
        plt.savefig('comparison_heat_simulation.png')
        print("Static plot saved to 'comparison_heat_simulation.png'")
