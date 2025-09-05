import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import time
from pandas import ExcelWriter

# ------------------------------
# Constants
E = 2.1e11
rho = 7850
g = 9.81
sigma_allow = 250e6
A_min = 0.0001
A_max = 0.01293
pop_size = 200
num_generations = 200   
# mutation_rate = 0.1
penalty_factor = 1e6
a = 19
n_runs = 1
F_de = 0.8      # عامل مقیاس‌دهی برای DE (Mutation factor)
CR_de = 0.9     # نرخ کراس‌اور برای DE
random.seed(42)
np.random.seed(42)
# ------------------------------

def get_load_vector(num_nodes):
    loads = np.zeros(num_nodes * 3)
    for i in range(num_nodes):
        if i == 0:
            mass = 2000
        elif 1 <= i <= 7:
            mass = 1500
        elif 8 <= i <= 12:
            mass = 1000
        elif 13 <= i <= 25:
            mass = 300
        elif 26 <= i <= 36:
            mass = 100
        else:
            mass = 0  # گره‌های 38 تا 49 مقید، بدون بار وزن
        loads[i * 3 + 2] = -mass * g
    # نیروی باد 50 نیوتن روی گره‌های 1 تا 37 در جهت X مثبت
    for i in range(37):
        loads[i * 3] += 50
    return loads

def bar_length(node1, node2):
    return np.linalg.norm(node1 - node2)

def assemble_stiffness_matrix(coords, areas, members, member_types):
    n_nodes = len(coords)
    K = np.zeros((n_nodes * 3, n_nodes * 3))
    for i, (start, end) in enumerate(members):
        n1, n2 = start - 1, end - 1
        x1, x2 = coords[n1], coords[n2]
        L = bar_length(x1, x2)
        a_vec = (x2 - x1) / L
        A = areas[member_types[i]]
        k_local = (A * E / L) * np.outer(a_vec, a_vec)
        dofs = [n1*3, n1*3+1, n1*3+2, n2*3, n2*3+1, n2*3+2]
        for ii in range(3):
            for jj in range(3):
                K[dofs[ii], dofs[jj]] += k_local[ii, jj]
                K[dofs[ii+3], dofs[jj+3]] += k_local[ii, jj]
                K[dofs[ii], dofs[jj+3]] -= k_local[ii, jj]
                K[dofs[ii+3], dofs[jj]] -= k_local[ii, jj]
    return K

def apply_boundary_conditions(K, F, fixed_dofs):
    for dof in sorted(fixed_dofs, reverse=True):
        K = np.delete(K, dof, axis=0)
        K = np.delete(K, dof, axis=1)
        F = np.delete(F, dof)
    return K, F

def compute_stress(coords, areas, members, displacements, member_types):
    stresses = np.zeros(len(members))
    for i, (start, end) in enumerate(members):
        n1, n2 = start - 1, end - 1
        x1, x2 = coords[n1], coords[n2]
        L = bar_length(x1, x2)
        a_vec = (x2 - x1) / L
        u1 = displacements[n1*3:n1*3+3]
        u2 = displacements[n2*3:n2*3+3]
        strain = np.dot(a_vec, u2 - u1) / L
        stresses[i] = E * strain
    return stresses

def calculate_fitness(individual, node_coords, members, member_types, num_movable_nodes):
    areas = individual[:a]
    moved_nodes = individual[a:].reshape(-1, 3)
    coords = node_coords.copy()
    coords[:num_movable_nodes] = moved_nodes
    weight = sum(rho * areas[member_types[i]] * bar_length(coords[s-1], coords[e-1])
                 for i, (s, e) in enumerate(members))
    K = assemble_stiffness_matrix(coords, areas, members, member_types)
    F = get_load_vector(len(coords))
    fixed_nodes = list(range(37, 49))
    fixed_dofs = [node*3 + d for node in fixed_nodes for d in range(3)]
    K_mod, F_mod = apply_boundary_conditions(K, F, fixed_dofs)
    try:
        U_mod = np.linalg.solve(K_mod, F_mod)
    except np.linalg.LinAlgError:
        return weight + penalty_factor * 100
    U = np.zeros(len(F))
    free_dofs = list(set(range(len(F))) - set(fixed_dofs))
    U[free_dofs] = U_mod
    stress = compute_stress(coords, areas, members, U, member_types)
    overstress_penalty = np.sum(np.abs(stress) > sigma_allow)
    return weight + penalty_factor * overstress_penalty

def initialize_population(n_move, coords):
    pop = []
    for _ in range(pop_size):
        A = np.random.uniform(A_min, A_max, a)
        delta = np.random.uniform(-0.02, 0.02, size=(n_move, 3))
        pos = coords[:n_move] + delta
        pop.append(np.concatenate([A, pos.flatten()]))
    return np.array(pop)

def differential_evolution(coords, members, member_types, num_movable_nodes):
    pop = initialize_population(num_movable_nodes, coords)
    best_fitness_history = []
    prev_best_fitness = None

    for gen in range(num_generations):
        fitness = np.array([calculate_fitness(ind, coords, members, member_types, num_movable_nodes) for ind in pop])
        idx = np.argmin(fitness)
        best_fitness = fitness[idx]
        best_fitness_history.append(best_fitness)

        prev_best_fitness = best_fitness

        new_pop = []
        for i in range(pop_size):
            idxs = list(range(pop_size))
            idxs.remove(i)
            a_, b_, c_ = random.sample(idxs, 3)
            x1, x2, x3 = pop[a_], pop[b_], pop[c_]
            mutant = x1 + F_de * (x2 - x3)
            mutant[:a] = np.clip(mutant[:a], A_min, A_max)
            trial = np.copy(pop[i])
            j_rand = random.randint(0, len(pop[i]) - 1)
            for j in range(len(pop[i])):
                if random.random() < CR_de or j == j_rand:
                    trial[j] = mutant[j]
            fit_target = calculate_fitness(pop[i], coords, members, member_types, num_movable_nodes)
            fit_trial = calculate_fitness(trial, coords, members, member_types, num_movable_nodes)
            if fit_trial < fit_target:
                new_pop.append(trial)
            else:
                new_pop.append(pop[i])
        pop = np.array(new_pop)

        print(f"Generation {gen+1}: Weight = {best_fitness:.2f} kg")

    plt.figure()
    plt.plot(best_fitness_history, 'b-', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (Weight in kg)')
    plt.title('Convergence Curve (DE)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    idx_best = np.argmin([calculate_fitness(ind, coords, members, member_types, num_movable_nodes) for ind in pop])
    return pop[idx_best]

def plot_truss(nodes, members, areas, member_types, title="Truss Structure", show_weights=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if show_weights:
        max_area = np.max(areas)
        for i, (start, end) in enumerate(members):
            p1 = nodes[start - 1]
            p2 = nodes[end - 1]
            xs, ys, zs = zip(p1, p2)
            lw = 1 + 10 * (areas[member_types[i]] / max_area)
            ax.plot(xs, ys, zs, color='b', linewidth=lw)
    else:
        for (start, end) in members:
            p1 = nodes[start - 1]
            p2 = nodes[end - 1]
            xs, ys, zs = zip(p1, p2)
            ax.plot(xs, ys, zs, color='gray', linewidth=1, linestyle='--')

    xs, ys, zs = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    ax.scatter(xs, ys, zs, color='k', s=10, label='All Nodes')

    ax.scatter(nodes[0, 0], nodes[0, 1], nodes[0, 2], color='yellow', s=70, marker='^', label='Load 2000 kg (Node 1)')
    ax.scatter(nodes[1:8, 0], nodes[1:8, 1], nodes[1:8, 2], color='green', s=50, marker='^', label='Load 1500 kg (Nodes 2-8)')
    ax.scatter(nodes[8:13, 0], nodes[8:13, 1], nodes[8:13, 2], color='#dfff00', s=50, marker='^', label='Load 1000 kg (Nodes 9-13)')
    ax.scatter(nodes[13:26, 0], nodes[13:26, 1], nodes[13:26, 2], color='#40e0d0', s=50, marker='^', label='Load 300 kg (Nodes 14-26)')
    ax.scatter(nodes[26:37, 0], nodes[26:37, 1], nodes[26:37, 2], color='blue', s=30, marker='^', label='Load 100 kg (Nodes 27-37)')
    ax.scatter(nodes[37:49, 0], nodes[37:49, 1], nodes[37:49, 2], color='navy', s=80, marker='s', label='Constrained Nodes 38-49 (No Load)')

    ax.scatter(nodes[0:37, 0], nodes[0:37, 1], nodes[0:37, 2], facecolors='none', edgecolors='red', s=90, marker='o', label='Wind Force 50 N (Nodes 1-37)')

    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()

def load_data():
    nodes = pd.read_excel("/home/sepehr/VS Code/Project/nodes_coordinates.xlsx")
    mems = pd.read_excel("/home/sepehr/VS Code/Project/truss_members.xlsx")
    return nodes[["X", "Y", "Z"]].to_numpy(), mems[["Start", "End"]].to_numpy()

if __name__ == "__main__":
    node_coords, members = load_data()
    member_types = np.array([i % a for i in range(len(members))])
    num_movable_nodes = 37

    plot_truss(node_coords, members, np.ones(len(members)), member_types,
               title="Initial Truss Structure (All Nodes and Members with Loads & Constraints)",
               show_weights=False)

    start_time = time.time()

    best_sol, best_fit = None, float("inf")
    for run in range(n_runs):
        print(f"\n===== Run {run+1} =====")
        sol = differential_evolution(node_coords.copy(), members, member_types, num_movable_nodes)
        fit = calculate_fitness(sol, node_coords.copy(), members, member_types, num_movable_nodes)
        print(f"Run {run+1} final fitness = {fit:.2f} kg")
        if fit < best_fit:
            best_sol, best_fit = sol.copy(), fit

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"\n⏱️ Total computation time: {elapsed_time:.2f} seconds")

    A_best = best_sol[:a]
    pos_best = best_sol[a:].reshape(-1, 3)
    original_coords = node_coords[:num_movable_nodes].copy()
    node_coords[:num_movable_nodes] = pos_best

    df_members = pd.DataFrame({
        "Start": members[:, 0],
        "End": members[:, 1],
        "Area (m^2)": [A_best[member_types[i]] for i in range(len(members))]
    })

    displacements = pos_best - original_coords
    df_disp = pd.DataFrame({
        "Node Index": np.arange(1, num_movable_nodes + 1),
        "X_Initial": original_coords[:, 0],
        "Y_Initial": original_coords[:, 1],
        "Z_Initial": original_coords[:, 2],
        "X_New": pos_best[:, 0],
        "Y_New": pos_best[:, 1],
        "Z_New": pos_best[:, 2],
        "dX": displacements[:, 0],
        "dY": displacements[:, 1],
        "dZ": displacements[:, 2],
        "Total_Displacement": np.linalg.norm(displacements, axis=1)
    })

    output_path = "/home/sepehr/VS Code/Project/D- DE_LoadSet2.xlsx"
    with ExcelWriter(output_path) as writer:
        df_members.to_excel(writer, sheet_name="Members_Areas", index=False)
        df_disp.to_excel(writer, sheet_name="Node_Displacements", index=False)

    print(f"\n✅ Final results saved to '{output_path}'")

    plot_truss(node_coords, members, A_best, member_types,
               title="Optimized Truss Structure", show_weights=True)
