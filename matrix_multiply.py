import numpy as np
import threading
import time
import csv
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MATRIX_SIZE   = 5000          # 5k x 5k
NUM_MATRICES  = 500           # number of random matrices
MAX_THREADS   = 10           # T=1 … T=8  (change to 2*cores if needed)

# One fixed constant matrix (shared, read-only)
CONSTANT_MATRIX = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)

results_lock = threading.Lock()


# ──────────────────────────────────────────────
# WORKER  – multiplies one random matrix
# ──────────────────────────────────────────────
def multiply_worker(index):
    random_matrix = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
    _ = np.dot(random_matrix, CONSTANT_MATRIX)   # actual multiplication


# ──────────────────────────────────────────────
# RUN with a given thread count, return seconds
# ──────────────────────────────────────────────
def run_with_threads(num_threads: int) -> float:
    print(f"\n▶  Running with {num_threads} thread(s)…")
    start = time.time()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(multiply_worker, i) for i in range(NUM_MATRICES)]
        for f in futures:
            f.result()          # wait for all

    elapsed = time.time() - start
    print(f"   ✔  Finished in {elapsed:.2f} s  ({elapsed/60:.2f} min)")
    return elapsed


# ──────────────────────────────────────────────
# PLOT
# ──────────────────────────────────────────────
def plot_results(thread_counts, times_sec):
    times_min = [t / 60 for t in times_sec]

    plt.figure(figsize=(8, 5))
    plt.plot(thread_counts, times_min, marker='o', color='steelblue', linewidth=2)
    plt.title("Execution Time vs Number of Threads")
    plt.xlabel("Number of Threads")
    plt.ylabel("Time Taken (min)")
    plt.xticks(thread_counts)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("execution_time_plot.png", dpi=150)
    plt.show()
    print("\n📊  Plot saved as execution_time_plot.png")


# ──────────────────────────────────────────────
# SAVE CSV
# ──────────────────────────────────────────────
def save_csv(thread_counts, times_sec):
    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Threads", "Time_sec", "Time_min"])
        for t, s in zip(thread_counts, times_sec):
            writer.writerow([t, round(s, 2), round(s / 60, 4)])
    print("📄  Results saved to results.csv")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    thread_counts = list(range(1, MAX_THREADS + 1))
    times_sec     = []

    for t in thread_counts:
        elapsed = run_with_threads(t)
        times_sec.append(elapsed)

    # ── Print table ──────────────────────────
    print("\n" + "=" * 55)
    print(f"{'Threads':<12}", end="")
    for t in thread_counts:
        print(f"T={t:<5}", end="")
    print()
    print(f"{'Time (min)':<12}", end="")
    for s in times_sec:
        print(f"{s/60:<7.2f}", end="")
    print()
    print("=" * 55)

    save_csv(thread_counts, times_sec)
    plot_results(thread_counts, times_sec)