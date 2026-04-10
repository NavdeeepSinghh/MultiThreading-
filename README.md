import numpy as np
import threading
import time
import csv
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════
MATRIX_SIZE  = 5000
NUM_MATRICES = 500
MAX_THREADS  = 10       # change to 2 * your core count if needed

# One fixed constant matrix (shared, read-only across all threads)
CONSTANT_MATRIX = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)

results_lock = threading.Lock()


# ══════════════════════════════════════════════════════════
# WORKER — multiplies one random matrix with constant matrix
# ══════════════════════════════════════════════════════════
def multiply_worker(index):
    random_matrix = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
    _ = np.dot(random_matrix, CONSTANT_MATRIX)


# ══════════════════════════════════════════════════════════
# RUN — executes all 500 multiplications with N threads
# ══════════════════════════════════════════════════════════
def run_with_threads(num_threads: int) -> float:
    print(f"\n{'═'*50}")
    print(f"  ▶  Starting with {num_threads} thread(s)...")
    print(f"{'═'*50}")

    start = time.time()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(multiply_worker, i) for i in range(NUM_MATRICES)]
        for i, f in enumerate(futures):
            f.result()
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start
                print(f"  ✔  {i+1}/{NUM_MATRICES} matrices done — {elapsed:.1f}s elapsed")

    total = time.time() - start
    print(f"\n  ✅  T={num_threads} COMPLETE → {total:.2f}s  ({total/60:.3f} min)")
    return total


# ══════════════════════════════════════════════════════════
# SAVE CSV
# ══════════════════════════════════════════════════════════
def save_csv(thread_counts, times_sec):
    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Threads", "Time_seconds", "Time_minutes"])
        for t, s in zip(thread_counts, times_sec):
            writer.writerow([t, round(s, 4), round(s / 60, 6)])

    print("\n📄  Results saved → results.csv")


# ══════════════════════════════════════════════════════════
# PRINT TABLE
# ══════════════════════════════════════════════════════════
def print_table(thread_counts, times_sec):
    print("\n")
    print("╔══════════════╦" + "═══════════╦" * (len(thread_counts) - 1) + "═══════════╗")

    header = "║   Threads    ║"
    for t in thread_counts:
        header += f"   T={t:<5}  ║"
    print(header)

    print("╠══════════════╬" + "═══════════╬" * (len(thread_counts) - 1) + "═══════════╣")

    row = "║ Time (min)   ║"
    for s in times_sec:
        row += f"  {s/60:>6.3f}   ║"
    print(row)

    print("╚══════════════╩" + "═══════════╩" * (len(thread_counts) - 1) + "═══════════╝")


# ══════════════════════════════════════════════════════════
# PLOT GRAPH
# ══════════════════════════════════════════════════════════
def plot_results(thread_counts, times_sec):
    times_min = [t / 60 for t in times_sec]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Multi-Threaded Matrix Multiplication — Performance Analysis",
                 fontsize=14, fontweight='bold', y=1.02)

    # ── Plot 1: Time in Minutes ──────────────────────────
    axes[0].plot(thread_counts, times_min,
                 marker='o', color='steelblue', linewidth=2.5,
                 markersize=8, markerfacecolor='white', markeredgewidth=2)
    axes[0].fill_between(thread_counts, times_min, alpha=0.1, color='steelblue')
    axes[0].set_title("Execution Time (Minutes)", fontweight='bold')
    axes[0].set_xlabel("Number of Threads")
    axes[0].set_ylabel("Time Taken (min)")
    axes[0].set_xticks(thread_counts)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Annotate each point
    for x, y in zip(thread_counts, times_min):
        axes[0].annotate(f"{y:.2f}m",
                         xy=(x, y), xytext=(0, 10),
                         textcoords='offset points',
                         ha='center', fontsize=8, color='steelblue')

    # ── Plot 2: Speedup vs T=1 ───────────────────────────
    baseline = times_sec[0]
    speedups = [baseline / t for t in times_sec]

    axes[1].bar(thread_counts, speedups,
                color=['#2ecc71' if s == max(speedups) else 'steelblue' for s in speedups],
                edgecolor='white', linewidth=0.5)
    axes[1].set_title("Speedup Relative to T=1", fontweight='bold')
    axes[1].set_xlabel("Number of Threads")
    axes[1].set_ylabel("Speedup (×)")
    axes[1].set_xticks(thread_counts)
    axes[1].grid(True, linestyle='--', alpha=0.5, axis='y')

    # Annotate bars
    for x, s in zip(thread_counts, speedups):
        axes[1].text(x, s + 0.02, f"{s:.2f}×",
                     ha='center', fontsize=8,
                     fontweight='bold' if s == max(speedups) else 'normal')

    plt.tight_layout()
    plt.savefig("execution_time_plot.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("📊  Plot saved → execution_time_plot.png")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("\n" + "█"*52)
    print("█   MULTI-THREADED MATRIX MULTIPLICATION         █")
    print(f"█   Matrix Size : {MATRIX_SIZE} x {MATRIX_SIZE}               █")
    print(f"█   Matrices    : {NUM_MATRICES}                              █")
    print(f"█   Max Threads : {MAX_THREADS}                               █")
    print("█"*52)

    thread_counts = list(range(1, MAX_THREADS + 1))
    times_sec     = []

    for t in thread_counts:
        elapsed = run_with_threads(t)
        times_sec.append(elapsed)
        print(f"\n  📸  >>> TAKE CPU SCREENSHOT NOW for T={t} <<<")
        time.sleep(3)   # 3 second pause to grab screenshot

    # ── Results ──────────────────────────────────────────
    print_table(thread_counts, times_sec)
    save_csv(thread_counts, times_sec)
    plot_results(thread_counts, times_sec)

    # ── Summary ──────────────────────────────────────────
    best_t    = thread_counts[times_sec.index(min(times_sec))]
    worst_t   = thread_counts[times_sec.index(max(times_sec))]
    print(f"\n  🏆  Best  performance : T={best_t}  ({min(times_sec)/60:.3f} min)")
    print(f"  🐢  Worst performance : T={worst_t} ({max(times_sec)/60:.3f} min)")
    print(f"  ⚡  Total speedup gained : {max(times_sec)/min(times_sec):.2f}×")
    print("\n  ✅  All done! Check execution_time_plot.png and results.csv\n")
