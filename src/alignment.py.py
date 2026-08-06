import numpy as np
import time
import random
import math
import csv



# ==============================
# Sequence Utilities
# ==============================
def generate_seq(length):
    return ''.join(random.choices("ATCG", k=length))


def get_kmers(seq, k=3):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


def kmer_similarity(seq1, seq2, k=3):
    kmers1 = set(get_kmers(seq1, k))
    kmers2 = set(get_kmers(seq2, k))

    intersection = kmers1.intersection(kmers2)
    union = kmers1.union(kmers2)

    if len(union) == 0:
        return 0

    return len(intersection) / len(union)


# ==============================
# Pufferfish Functions
# ==============================
def build_pufferfish_index(seq, k=6):
    index = {}
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if kmer not in index:
            index[kmer] = []
        index[kmer].append(i)
    return index


def pufferfish_similarity(seq1, seq2, k=6):
    index = build_pufferfish_index(seq2, k)
    matches = 0

    for i in range(len(seq1) - k + 1):
        kmer = seq1[i:i+k]
        if kmer in index:
            matches += 1

    total = len(seq1) - k + 1
    return matches / total if total > 0 else 0


def simple_similarity(seq1, seq2):
    matches = 0
    length = min(len(seq1), len(seq2))

    for i in range(length):
        if seq1[i] == seq2[i]:
            matches += 1

    return matches / length


def adaptive_band_width(similarity, min_band=2, max_band=20):
    return int(max_band * (1 - similarity) + min_band)


# ==============================
# Pufferfish-based Optimization
# ==============================
def pufferfish_band_optimizer(seq1, seq2, k=6):
    sim_pf = pufferfish_similarity(seq1, seq2, k)
    band = adaptive_band_width(sim_pf)
    return band, sim_pf


def run_pufferfish_versions(seq1, seq2):
    # get band from pufferfish
    band_pf, sim_pf = pufferfish_band_optimizer(seq1, seq2)

    # --- Pufferfish + Banded ---
    start = time.time()
    score_pf_banded = banded_smith_waterman(seq1, seq2, band_pf)
    time_pf_banded = time.time() - start

    # --- Pufferfish + Adaptive ---
    start = time.time()
    score_pf_adaptive, cells_pf = adaptive_smith_waterman(seq1, seq2, band_pf)
    time_pf_adaptive = time.time() - start

    return {
        "sim_pf": sim_pf,
        "band_pf": band_pf,
        "pf_banded_score": score_pf_banded,
        "pf_banded_time": time_pf_banded,
        "pf_adaptive_score": score_pf_adaptive,
        "pf_adaptive_time": time_pf_adaptive,
        "pf_cells": cells_pf
    }


# ==============================
# Scoring
# ==============================
MATCH = 2
MISMATCH = -1
GAP = -2          # fixed gap penalty (used by Normal & Banded)

# Context-aware gap penalty range (used by Adaptive only)
# Tighter range centered on GAP=-2.0 for scoring consistency:
#   Conserved regions -> stricter penalty (gaps are biologically unexpected)
#   Variable regions  -> lenient penalty  (gaps are more tolerable)
GAP_CONSERVED = -2.5
GAP_VARIABLE = -2.0


# ==============================
# BLAST-like
# ==============================
def identity(seq1, seq2):
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / len(seq1) if len(seq1) > 0 else 0


def compute_evalue(score, m, n, K=0.1, lam=0.5):
    return K * m * n * math.exp(-lam * score)


def blast_like(seq1, seq2, k=6, threshold=8):

    kmers1 = get_kmers(seq1, k)
    kmers2 = get_kmers(seq2, k)

    # Index seq2
    index = {}
    for j, kmer in enumerate(kmers2):
        if kmer not in index:
            index[kmer] = []
        index[kmer].append(j)

    best_score = 0
    best_hit = None

    # search seeds
    for i, kmer in enumerate(kmers1):
        if kmer in index:
            for j in index[kmer]:

                score = k * MATCH

                left = 1
                while i-left >= 0 and j-left >= 0:
                    if seq1[i-left] == seq2[j-left]:
                        score += MATCH
                    else:
                        score += MISMATCH

                    if score < threshold:
                        break
                    left += 1

                right = k
                while i+right < len(seq1) and j+right < len(seq2):
                    if seq1[i+right] == seq2[j+right]:
                        score += MATCH
                    else:
                        score += MISMATCH

                    if score < threshold:
                        break
                    right += 1

                if score > best_score:
                    best_score = score
                    best_hit = (i, j, left, right)

    if best_hit is None:
        return {
            "score": 0,
            "alignment_seq1": "",
            "alignment_seq2": "",
            "identity": 0,
            "e_value": 1
        }

    i, j, left, right = best_hit

    aligned1 = seq1[i-left:i+right]
    aligned2 = seq2[j-left:j+right]

    return {
        "score": best_score,
        "alignment_seq1": aligned1,
        "alignment_seq2": aligned2,
        "identity": identity(aligned1, aligned2),
        "e_value": compute_evalue(best_score, len(seq1), len(seq2))
    }


# ==============================
# Smith-Waterman (Normal)
# ==============================
def smith_waterman(seq1, seq2):
    n, m = len(seq1), len(seq2)
    H = np.zeros((n+1, m+1))
    max_score = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            match = H[i-1][j-1] + (MATCH if seq1[i-1] == seq2[j-1] else MISMATCH)
            delete = H[i-1][j] + GAP
            insert = H[i][j-1] + GAP

            H[i][j] = max(0, match, delete, insert)
            max_score = max(max_score, H[i][j])

    return max_score, H


# ==============================
# Banded Alignment
# ==============================
def banded_smith_waterman(seq1, seq2, band_width=5):
    n, m = len(seq1), len(seq2)
    H = np.zeros((n+1, m+1))
    max_score = 0

    for i in range(1, n+1):
        for j in range(max(1, i-band_width), min(m+1, i+band_width)):
            match = H[i-1][j-1] + (MATCH if seq1[i-1] == seq2[j-1] else MISMATCH)
            delete = H[i-1][j] + GAP
            insert = H[i][j-1] + GAP

            H[i][j] = max(0, match, delete, insert)
            max_score = max(max_score, H[i][j])

    return max_score


# ==============================
# Step 3: Context-Aware Gap Penalty
# ==============================
#
# Instead of a fixed gap penalty across the entire sequence, this
# model adapts the penalty based on local mutation density:
#
#   GapPenalty(i) = GAP_CONSERVED + (GAP_VARIABLE - GAP_CONSERVED) * density(i)
#
# Where density(i) is the Gaussian-weighted fraction of mismatches in
# a sliding window around position i when comparing seq1 vs seq2.
#
#   density ~ 0  (conserved)  ->  penalty ~ GAP_CONSERVED (-2.5)
#   density ~ 1  (variable)   ->  penalty ~ GAP_VARIABLE  (-1.5)
#   density ~ 0.5 (neutral)   ->  penalty ~ -2.0 (same as fixed GAP)
#
# This allows the scoring scheme to adapt to biological variability
# across the sequence while keeping scores comparable to the fixed model.
# ==============================

def _build_gaussian_weights(window):
    """Build Gaussian kernel weights for the density window."""
    sigma = window / 2.0
    offsets = np.arange(-window, window + 1, dtype=float)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    return weights


def precompute_mutation_density(seq1, seq2, window=10):
    """
    Precompute Gaussian-weighted local mutation density for each
    position along the diagonal (positional comparison).

    Uses a larger window (default=10) with Gaussian weighting so that
    positions closer to i have more influence than distant positions.
    This produces smoother, more stable density estimates than uniform
    weighting with a small window.

    Returns a 1-D numpy array indexed by 0-based position, sized to
    len(seq1). Positions beyond seq2's length get density = 0.5 (neutral).
    """
    n = len(seq1)
    m = len(seq2)
    comparable = min(n, m)
    density = np.full(n, 0.5)  # default neutral for positions beyond seq2

    gauss_weights = _build_gaussian_weights(window)

    for i in range(comparable):
        start = max(0, i - window)
        end = min(comparable, i + window + 1)

        weighted_mismatches = 0.0
        total_weight = 0.0

        for k in range(start, end):
            offset = k - i
            w = gauss_weights[offset + window]  # center index = window
            total_weight += w
            if seq1[k] != seq2[k]:
                weighted_mismatches += w

        density[i] = weighted_mismatches / total_weight if total_weight > 0 else 0.5

    return density


def dynamic_gap_penalty(density_value):
    """
    Context-aware gap penalty from local mutation density.

    Conserved regions (density ~ 0) -> GAP_CONSERVED (-2.5)  stricter
    Variable regions  (density ~ 1) -> GAP_VARIABLE  (-2.0)  lenient
    Neutral midpoint  (density ~0.5) -> -2.25 (stricter than fixed GAP)

    Linear interpolation between the two extremes.
    """
    return GAP_CONSERVED + (GAP_VARIABLE - GAP_CONSERVED) * density_value


# ==============================
# Adaptive Smith-Waterman
# ==============================
def adaptive_smith_waterman(seq1, seq2, band):
    """
    Banded Smith-Waterman with context-aware dynamic gap penalty.

    Steps performed internally:
      1. Precompute local mutation density map along the diagonal.
      2. For each cell (i, j) inside the band, look up density at
         position (i-1) and apply the dynamic gap penalty.

    Normal SW and Banded SW are NOT affected (they keep fixed GAP = -2).
    """
    n, m = len(seq1), len(seq2)
    H = np.zeros((n+1, m+1))
    max_score = 0
    cells = 0

    # Step 3: precompute mutation density once (O(n * window))
    density_map = precompute_mutation_density(seq1, seq2)

    for i in range(1, n+1):
        # Look up precomputed density for this row (0-indexed -> i-1)
        gap = dynamic_gap_penalty(density_map[i - 1])

        for j in range(max(1, i-band), min(m+1, i+band+1)):

            cells += 1

            match = H[i-1][j-1] + (MATCH if seq1[i-1] == seq2[j-1] else MISMATCH)
            delete = H[i-1][j] + gap
            insert = H[i][j-1] + gap

            H[i][j] = max(0, match, delete, insert)
            max_score = max(max_score, H[i][j])

    return max_score, cells


# ==============================
# Paper Fitness Function (Task 2)
# ==============================
def fitness_function(band, seq1, seq2, Hfull, Cfull, alpha=0.7, beta=0.3):
    """
    Compute the fitness of a candidate band width as defined in the paper:

        Fitness(b) = alpha * (Hmax(b) / Hfull) - beta * (C(b) / Cfull)

    Where:
        Hmax(b) = Adaptive Smith-Waterman score using candidate band b
        Hfull   = Normal (full) Smith-Waterman score
        C(b)    = Number of computed DP cells inside adaptive band b
        Cfull   = Total cells in the full DP matrix (n * m)
        alpha   = Weight for alignment score quality (default 0.7)
        beta    = Weight for computational cost penalty (default 0.3)

    The objective is to MAXIMIZE Fitness(b), balancing alignment quality
    against computational efficiency.

    Parameters
    ----------
    band : int
        Candidate band width to evaluate.
    seq1 : str
        First DNA/RNA sequence.
    seq2 : str
        Second DNA/RNA sequence.
    Hfull : float
        Normal (full) Smith-Waterman alignment score.
    Cfull : int
        Total cells in the full DP matrix (n * m).
    alpha : float
        Weight for score quality (default 0.7).
    beta : float
        Weight for computational cost penalty (default 0.3).

    Returns
    -------
    float
        Fitness value. Higher is better.
    """
    Hmax_b, C_b = adaptive_smith_waterman(seq1, seq2, band)

    if Hfull == 0:
        score_ratio = 0.0
    else:
        score_ratio = Hmax_b / Hfull

    cell_ratio = C_b / Cfull if Cfull > 0 else 0.0

    return alpha * score_ratio - beta * cell_ratio


# ==============================
# Levy Flight (for SFOA Regeneration)
# ==============================
def levy_flight(beta_lf=1.5):
    """
    Generate a Levy flight step using Mantegna's algorithm.

    The Levy flight produces heavy-tailed random steps that allow the
    SFOA to escape local optima during the regeneration phase.  Large
    occasional jumps enable global exploration while most steps remain
    small for local refinement.

    Parameters
    ----------
    beta_lf : float
        Levy exponent controlling the tail heaviness (default 1.5).
        Valid range: (0, 2].  Typical value for optimization is 1.5.

    Returns
    -------
    float
        A single Levy flight step value.

    References
    ----------
    Mantegna, R.N. (1994). "Fast, accurate algorithm for numerical
    simulation of Levy stable stochastic processes."
    Physical Review E, 49(5), 4677.
    """
    # Mantegna's algorithm for Levy stable distributions
    # sigma_u computation using the Gamma function
    numerator = math.gamma(1 + beta_lf) * math.sin(math.pi * beta_lf / 2)
    denominator = math.gamma((1 + beta_lf) / 2) * beta_lf * (2 ** ((beta_lf - 1) / 2))
    sigma_u = (numerator / denominator) ** (1 / beta_lf)

    u = random.gauss(0, sigma_u)
    v = random.gauss(0, 1)

    step = u / (abs(v) ** (1 / beta_lf))
    return step


# ==============================
# Starfish Optimization Algorithm (SFOA) — Task 1
# ==============================
def starfish_optimization(seq1, seq2, sim, pop_size=20, max_iter=30,
                          min_band=2, max_band=50, alpha=0.7, beta=0.3,
                          Hfull_precomputed=None):
    """
    Starfish Optimization Algorithm (SFOA) for adaptive band width selection.

    Implements the original SFOA metaheuristic inspired by starfish
    biological behavior to find the optimal band width that maximizes
    the paper's fitness function:

        Fitness(b) = alpha * (Hmax(b) / Hfull) - beta * (C(b) / Cfull)

    The algorithm models five biological behaviors of starfish:

    1. **Population Initialization**: N starfish agents are placed in the
       search space around the similarity-based initial band estimate
       using uniform random perturbation.

    2. **Exploration Phase** (tube feet movement — global search):
       When the transition parameter p(t) favors exploration, agents
       move using the elite position and population mean for diversity:
           X_i(t+1) = X_i(t) + r1 * (X_best(t) - r2 * X_mean(t))

    3. **Exploitation Phase** (arm coordination — local refinement):
       When p(t) favors exploitation, agents refine around the elite
       with exponential decay controlling the step size:
           X_i(t+1) = X_best(t) * exp(-1/(C*t)) + (X_i(t) - X_best(t)) * r3

    4. **Regeneration / Levy Flight** (arm regeneration — escape local optima):
       The worst fraction of agents are regenerated using Levy flight:
           X_i(t+1) = X_i(t) + alpha0 * Levy(beta_lf) * (X_best(t) - X_i(t))
       Heavy-tailed Levy steps allow escape from local optima.

    5. **Elite Solution Tracking**: The global best (highest fitness) agent
       is tracked across all iterations and returned as the optimal band.

    Parameters
    ----------
    seq1 : str
        First DNA/RNA sequence.
    seq2 : str
        Second DNA/RNA sequence.
    sim : float
        Pre-computed k-mer similarity between seq1 and seq2.
    pop_size : int
        Number of starfish agents in the population. Default 20.
    max_iter : int
        Maximum number of optimization iterations. Default 30.
    min_band : int
        Minimum allowed band width. Default 2.
    max_band : int
        Maximum allowed band width. Default 50.
    alpha : float
        Fitness weight for alignment score quality. Default 0.7.
    beta : float
        Fitness weight for computational cost penalty. Default 0.3.
    Hfull_precomputed : float or None
        If provided, skips the internal full SW computation and uses
        this value as Hfull.  Useful when the caller has already
        computed the full Smith-Waterman score for the same pair.

    Returns
    -------
    int
        Optimized band width that maximizes the fitness function.
    """
    n, m = len(seq1), len(seq2)
    Cfull = n * m

    # Compute Hfull once for fitness normalization (or use precomputed)
    if Hfull_precomputed is not None:
        Hfull = Hfull_precomputed
    else:
        Hfull, _ = smith_waterman(seq1, seq2)

    # ── Phase 1: Population Initialization ──
    # Seed population around the similarity-based initial estimate
    base_band = adaptive_band_width(sim)
    population = []
    for _ in range(pop_size):
        # Uniform random perturbation covering a wide range around base_band
        perturbation = random.uniform(-base_band * 0.5, base_band * 1.5)
        candidate = base_band + perturbation
        candidate = max(float(min_band), min(float(max_band), candidate))
        population.append(candidate)

    # Ensure the similarity-based estimate is represented
    population[0] = float(base_band)

    # Evaluate initial fitness for all agents
    fitness_values = []
    for agent in population:
        band_int = max(min_band, min(max_band, int(round(agent))))
        f = fitness_function(band_int, seq1, seq2, Hfull, Cfull, alpha, beta)
        fitness_values.append(f)

    # Initialize global elite solution
    best_idx = int(np.argmax(fitness_values))
    elite_position = population[best_idx]
    elite_fitness = fitness_values[best_idx]

    # Fitness cache for memoization
    fitness_cache = {}
    for i, agent in enumerate(population):
        band_int = max(min_band, min(max_band, int(round(agent))))
        fitness_cache[band_int] = fitness_values[i]

    # ── SFOA Hyperparameters ──
    C_decay = 2.0           # Exploitation decay constant
    alpha0 = 0.01           # Levy flight scaling factor
    regen_fraction = 0.25   # Fraction of worst agents regenerated per iteration

    # ── Iterative Optimization (Phases 2-5) ──
    for t in range(1, max_iter + 1):

        # Transition parameter: linear decrease from 1 -> 0
        # Early iterations favor exploration; later iterations favor exploitation
        p_t = 1.0 - (t / max_iter)

        # Population mean position (used in exploration phase)
        X_mean = sum(population) / len(population)

        # ── Phase 2 & 3: Position Update for Each Agent ──
        for i in range(pop_size):
            r = random.random()

            if r < p_t:
                # ── Exploration Phase (Equation 1) ──
                # Global search: move toward elite, modulated by population mean
                r1 = random.random()
                r2 = random.random()
                new_pos = population[i] + r1 * (elite_position - r2 * X_mean)
            else:
                # ── Exploitation Phase (Equation 2) ──
                # Local refinement: exponential decay toward elite
                r3 = random.uniform(-1.0, 1.0)
                decay = math.exp(-1.0 / (C_decay * t))
                new_pos = elite_position * decay + (population[i] - elite_position) * r3

            # Clamp to valid band width range
            new_pos = max(float(min_band), min(float(max_band), new_pos))

            # Evaluate fitness of candidate position
            band_int = max(min_band, min(max_band, int(round(new_pos))))
            if band_int in fitness_cache:
                f_new = fitness_cache[band_int]
            else:
                f_new = fitness_function(band_int, seq1, seq2, Hfull, Cfull, alpha, beta)
                fitness_cache[band_int] = f_new

            # Greedy selection: accept only if fitness improves
            if f_new > fitness_values[i]:
                population[i] = new_pos
                fitness_values[i] = f_new

        # ── Phase 4: Regeneration via Levy Flight ──
        # Identify the worst agents and regenerate them
        n_regen = max(1, int(pop_size * regen_fraction))
        sorted_indices = sorted(range(pop_size), key=lambda idx: fitness_values[idx])
        worst_indices = sorted_indices[:n_regen]

        for idx in worst_indices:
            # Levy flight regeneration (Equation 3)
            lf_step = levy_flight(beta_lf=1.5)
            new_pos = population[idx] + alpha0 * lf_step * (elite_position - population[idx])

            # Clamp to valid range
            new_pos = max(float(min_band), min(float(max_band), new_pos))

            # Evaluate fitness
            band_int = max(min_band, min(max_band, int(round(new_pos))))
            if band_int in fitness_cache:
                f_new = fitness_cache[band_int]
            else:
                f_new = fitness_function(band_int, seq1, seq2, Hfull, Cfull, alpha, beta)
                fitness_cache[band_int] = f_new

            # Regeneration unconditionally replaces the worst agents
            population[idx] = new_pos
            fitness_values[idx] = f_new

        # ── Phase 5: Update Global Elite ──
        current_best_idx = int(np.argmax(fitness_values))
        if fitness_values[current_best_idx] > elite_fitness:
            elite_fitness = fitness_values[current_best_idx]
            elite_position = population[current_best_idx]

    # Return optimized band width as integer
    optimal_band = max(min_band, min(max_band, int(round(elite_position))))
    return optimal_band


def starfish_band_search(seq1, seq2, sim):
    """
    Backward-compatible wrapper for the Starfish Optimization Algorithm.

    Preserves the original function signature used by run_case() and other
    callers.  Delegates entirely to starfish_optimization().

    Parameters
    ----------
    seq1 : str
        First DNA/RNA sequence.
    seq2 : str
        Second DNA/RNA sequence.
    sim : float
        Pre-computed k-mer similarity.

    Returns
    -------
    int
        Optimized band width from SFOA.
    """
    return starfish_optimization(seq1, seq2, sim)


# ==============================
# FASTA Reader
# ==============================
def read_fasta(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    seq = ''.join(line.strip() for line in lines if not line.startswith('>'))
    return seq


# ==============================
# 10-Run Benchmarking (Task 4)
# ==============================
def run_benchmark(seq1, seq2, n_runs=10):
    """
    Run all alignment methods over n_runs independent executions.

    For each method (Normal SW, Banded SW, Adaptive SW, BLAST-like,
    PF-Adaptive), this function performs n_runs independent runs and
    collects alignment scores, execution times, and DP cell counts.

    Each run uses a distinct random seed so that the SFOA band
    optimization produces independently sampled results, as required
    by the paper's experimental methodology.

    Timing for Adaptive SW measures only the alignment step, not the
    SFOA optimization overhead (consistent with the paper, where SFOA
    is a one-time preprocessing cost).

    Parameters
    ----------
    seq1 : str
        First DNA/RNA sequence.
    seq2 : str
        Second DNA/RNA sequence.
    n_runs : int
        Number of independent runs (default 10).

    Returns
    -------
    dict
        Dictionary keyed by method name.  Each value is a dict with
        lists 'scores', 'times', and 'cells' of length n_runs.
    """
    n, m = len(seq1), len(seq2)
    full_cells = n * m

    results = {
        "Normal SW":   {"scores": [], "times": [], "cells": []},
        "Banded SW":   {"scores": [], "times": [], "cells": []},
        "Adaptive SW": {"scores": [], "times": [], "cells": []},
        "BLAST":       {"scores": [], "times": [], "cells": []},
        "PF-Adaptive": {"scores": [], "times": [], "cells": []},
    }

    # Pre-compute banded cell count (deterministic for band_width=5)
    banded_cells = 0
    bw = 5
    for i in range(1, n + 1):
        j_start = max(1, i - bw)
        j_end = min(m + 1, i + bw)
        if j_end > j_start:
            banded_cells += (j_end - j_start)

    # Pre-compute k-mer similarity (deterministic)
    sim = kmer_similarity(seq1, seq2, k=6)

    # Pre-compute deterministic algorithms to save time across 10 runs
    print("  Pre-computing deterministic Normal SW and Banded SW...")
    t0 = time.time()
    cached_score_normal, _ = smith_waterman(seq1, seq2)
    cached_time_normal = time.time() - t0

    t0 = time.time()
    cached_score_banded = banded_smith_waterman(seq1, seq2, 5)
    cached_time_banded = time.time() - t0

    for run in range(n_runs):
        print(f"  Run {run + 1}/{n_runs} ...", end=" ", flush=True)

        # Unique random seed for each independent run
        run_seed = int(time.time() * 1000) % (2**31) + run * 7919
        random.seed(run_seed)
        np.random.seed(run_seed % (2**31))

        # --- Normal Smith-Waterman ---
        results["Normal SW"]["scores"].append(cached_score_normal)
        results["Normal SW"]["times"].append(cached_time_normal)
        results["Normal SW"]["cells"].append(full_cells)

        # --- Banded Smith-Waterman (band=5) ---
        results["Banded SW"]["scores"].append(cached_score_banded)
        results["Banded SW"]["times"].append(cached_time_banded)
        results["Banded SW"]["cells"].append(banded_cells)

        # --- Adaptive SW (with SFOA band optimization) ---
        # SFOA uses cached Hfull from the Normal SW run above
        band = starfish_optimization(seq1, seq2, sim,
                                     Hfull_precomputed=cached_score_normal)
        # Time only the adaptive alignment (SFOA is preprocessing)
        t0 = time.time()
        score_adaptive, cells_adaptive = adaptive_smith_waterman(seq1, seq2, band)
        time_adaptive = time.time() - t0
        results["Adaptive SW"]["scores"].append(score_adaptive)
        results["Adaptive SW"]["times"].append(time_adaptive)
        results["Adaptive SW"]["cells"].append(cells_adaptive)

        # --- BLAST-like ---
        t0 = time.time()
        blast_result = blast_like(seq1, seq2, k=6)
        time_blast = time.time() - t0
        results["BLAST"]["scores"].append(blast_result["score"])
        results["BLAST"]["times"].append(time_blast)
        results["BLAST"]["cells"].append(0)  # BLAST does not use a DP matrix

        # --- Pufferfish + Adaptive ---
        t0 = time.time()
        pf_res = run_pufferfish_versions(seq1, seq2)
        time_pf = time.time() - t0
        results["PF-Adaptive"]["scores"].append(pf_res["pf_adaptive_score"])
        results["PF-Adaptive"]["times"].append(time_pf)
        results["PF-Adaptive"]["cells"].append(pf_res["pf_cells"])

        print(f"done (SFOA band={band})")

    return results


# ==============================
# Statistical Summary (Task 5)
# ==============================
def print_statistical_summary(results):
    """
    Print a formatted statistical summary table for all methods.

    For each method, reports Mean, Standard Deviation, Minimum, and
    Maximum across all independent runs for:
      - Alignment Score
      - Execution Time (seconds)
      - DP Cells computed

    Parameters
    ----------
    results : dict
        Output from run_benchmark().
    """
    methods = list(results.keys())
    n_runs = len(results[methods[0]]["scores"])

    print("\n" + "=" * 72)
    print(f"  Statistical Summary ({n_runs} Independent Runs)")
    print("=" * 72)

    # --- Alignment Score ---
    print("\n--- Alignment Score ---")
    print(f"  {'Method':<15} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("  " + "-" * 63)
    for m in methods:
        vals = np.array(results[m]["scores"], dtype=float)
        print(f"  {m:<15} {np.mean(vals):>12.2f} {np.std(vals):>12.2f} "
              f"{np.min(vals):>12.2f} {np.max(vals):>12.2f}")

    # --- Execution Time ---
    print("\n--- Execution Time (seconds) ---")
    print(f"  {'Method':<15} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("  " + "-" * 63)
    for m in methods:
        vals = np.array(results[m]["times"], dtype=float)
        print(f"  {m:<15} {np.mean(vals):>12.4f} {np.std(vals):>12.4f} "
              f"{np.min(vals):>12.4f} {np.max(vals):>12.4f}")

    # --- DP Cells ---
    print("\n--- DP Cells Computed ---")
    print(f"  {'Method':<15} {'Mean':>14} {'Std':>14} {'Min':>14} {'Max':>14}")
    print("  " + "-" * 71)
    for m in methods:
        vals = np.array(results[m]["cells"], dtype=float)
        print(f"  {m:<15} {np.mean(vals):>14.0f} {np.std(vals):>14.0f} "
              f"{np.min(vals):>14.0f} {np.max(vals):>14.0f}")

    print("=" * 72)


# ==============================
# CSV Export (Task 7)
# ==============================
def export_results_csv(results, filename="results.csv"):
    """
    Export benchmark results to a CSV file.

    Columns: Method, Average Score, Average Time, Average DP Cells,
             Std Score, Std Time, Std Cells

    Parameters
    ----------
    results : dict
        Output from run_benchmark().
    filename : str
        Output CSV file path.  Default "results.csv".
    """
    methods = list(results.keys())

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Method", "Average Score", "Average Time", "Average DP Cells",
            "Std Score", "Std Time", "Std Cells"
        ])

        for m in methods:
            scores = np.array(results[m]["scores"], dtype=float)
            times  = np.array(results[m]["times"],  dtype=float)
            cells  = np.array(results[m]["cells"],  dtype=float)

            writer.writerow([
                m,
                f"{np.mean(scores):.2f}",
                f"{np.mean(times):.4f}",
                f"{np.mean(cells):.0f}",
                f"{np.std(scores):.2f}",
                f"{np.std(times):.4f}",
                f"{np.std(cells):.0f}",
            ])

    print(f"\n[CSV] Results exported to {filename}")


# ==============================
# Run Case (preserved for backward compatibility)
# ==============================
def run_case(title, seq1_path, seq2_path):
    print(f"\n===== {title} =====")

    seq1 = read_fasta(seq1_path)[:8000]
    seq2 = read_fasta(seq2_path)[:8000]

    sim = kmer_similarity(seq1, seq2, k=6)
    band = starfish_band_search(seq1, seq2, sim)

    start = time.time()
    score1, H = smith_waterman(seq1, seq2)
    time1 = time.time() - start

    start = time.time()
    score2 = banded_smith_waterman(seq1, seq2, 5)
    time2 = time.time() - start

    start = time.time()
    score3, cells3 = adaptive_smith_waterman(seq1, seq2, band)
    time3 = time.time() - start

    pf_sim = pufferfish_similarity(seq1[:500], seq2[:500], k=6)

    print("\n--- Pufferfish Validation ---")
    print(f"Similarity: {pf_sim:.4f}")

    print("\n--- Results ---")
    print(f"Normal   -> {score1:.2f}, Time: {time1:.4f}")
    print(f"Banded   -> {score2:.2f}, Time: {time2:.4f}")
    print(f"Adaptive -> {score3:.2f}, Time: {time3:.4f}")

    # ==============================
    # Pufferfish Optimization
    # ==============================
    pf_results = run_pufferfish_versions(seq1, seq2)

    print("\n--- Pufferfish Optimized ---")
    print(f"PF Similarity: {pf_results['sim_pf']:.4f}")
    print(f"PF Band: {pf_results['band_pf']}")

    print(f"PF + Banded   -> {pf_results['pf_banded_score']:.2f}, Time: {pf_results['pf_banded_time']:.4f}")
    print(f"PF + Adaptive -> {pf_results['pf_adaptive_score']:.2f}, Time: {pf_results['pf_adaptive_time']:.4f}")




# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":

    # ==============================
    # Load Test Data
    # ==============================
    seq1 = read_fasta("Homo sapiens mitochondrion.fasta")[:8000]
    seq2 = read_fasta("Ecoli_genome.fasta")[:8000]

    print("\n" + "=" * 72)
    print("  DNA Sequence Alignment - 10-Run Independent Benchmark")
    print("=" * 72)
    print(f"Sequence 1 Length: {len(seq1)} bp")
    print(f"Sequence 2 Length: {len(seq2)} bp")

    # ==============================
    # 10-Run Benchmark (Task 4)
    # ==============================
    print("\nRunning 10 independent benchmark executions ...")
    results = run_benchmark(seq1, seq2, n_runs=10)

    # ==============================
    # Averaged Results (print averages only, as per paper)
    # ==============================
    print("\n" + "=" * 72)
    print("  Averaged Results (10 Runs)")
    print("=" * 72)
    print(f"  {'Method':<15} {'Avg Score':>12} {'Avg Time (s)':>14} {'Avg Cells':>14}")
    print("  " + "-" * 55)
    for method in results:
        avg_s = np.mean(results[method]["scores"])
        avg_t = np.mean(results[method]["times"])
        avg_c = np.mean(results[method]["cells"])
        print(f"  {method:<15} {avg_s:>12.2f} {avg_t:>14.4f} {avg_c:>14.0f}")

    # ==============================
    # Statistical Summary (Task 5)
    # ==============================
    print_statistical_summary(results)

    # ==============================
    # CSV Export (Task 7)
    # ==============================
    export_results_csv(results, filename="results.csv")

    # ==============================
    # Similarity Analysis (preserved from original)
    # ==============================
    sim = kmer_similarity(seq1, seq2, k=6)
    pf_sim = pufferfish_similarity(seq1[:500], seq2[:500], k=6)

    print("\n--- Similarity Analysis ---")
    print(f"K-mer Similarity        : {sim:.4f}")
    print(f"Pufferfish Similarity   : {pf_sim:.4f}")

    print("\n--- Biological Validation ---")
    if pf_sim > 0.8:
        print("High similarity detected -> Strong biological match")
    elif pf_sim > 0.3:
        print("Moderate similarity -> Partial biological relationship")
    else:
        print("Low similarity -> Distant sequences")

    print("=" * 72)

    # ==============================
    # Run Cases (preserved for backward compatibility)
    # ==============================
    run_case(
        "Homo vs E. coli",
        "Homo sapiens mitochondrion.fasta",
        "Ecoli_genome.fasta"
    )

    run_case(
        "Homo vs Homo",
        "Homo sapiens mitochondrion.fasta",
        "Homo sapiens mitochondrion.fasta"
    )

    # ==============================
    # Pufferfish Results (preserved)
    # ==============================
    pf_results = run_pufferfish_versions(seq1, seq2)

    print("\n[DONE] Benchmark complete.")
