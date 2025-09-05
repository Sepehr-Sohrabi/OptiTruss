import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# Constants
E = 2.1e11
sigma_allow = 250e6
g = 9.81
a = 19
wind_force = 50
increment = 10  # Step for increasing load (N)
max_nodes_with_load = 37
# ------------------------------

def load_data():
    nodes = pd.read_excel("/home/sepehr/VS Code/Project/nodes_coordinates.xlsx")
    mems = pd.read_excel("/home/sepehr/VS Code/Project/truss_members.xlsx")
    design = pd.read_excel("/home/sepehr/VS Code/Project/C- GA_LoadSet2.xlsx", sheet_name="Members_Areas")
    displ = pd.read_excel("/home/sepehr/VS Code/Project/C- GA_LoadSet2.xlsx", sheet_name="Node_Displacements")
    return (
        nodes[["X", "Y", "Z"]].to_numpy(),
        mems[["Start", "End"]].to_numpy(),
        design["Area (m^2)"].to_numpy(),
        displ[["X_New", "Y_New", "Z_New"]].to_numpy()
    )

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

def get_max_load_per_node(coords, members, areas, member_types):
    num_nodes = len(coords)
    fixed_nodes = list(range(37, 49))
    fixed_dofs = [node*3 + d for node in fixed_nodes for d in range(3)]

    for trial_load in range(0, 100000, increment):
        F = np.zeros(num_nodes * 3)
        for i in range(max_nodes_with_load):
            F[i*3 + 2] = -trial_load
            F[i*3] += wind_force
        K = assemble_stiffness_matrix(coords, areas, members, member_types)
        K_mod, F_mod = apply_boundary_conditions(K, F, fixed_dofs)
        try:
            U_mod = np.linalg.solve(K_mod, F_mod)
        except np.linalg.LinAlgError:
            break
        U = np.zeros(len(F))
        free_dofs = list(set(range(len(F))) - set(fixed_dofs))
        U[free_dofs] = U_mod
        stress = compute_stress(coords, areas, members, U, member_types)
        if np.any(np.abs(stress) > sigma_allow):
            return trial_load - increment  # Last valid load
    return trial_load

def plot_truss(nodes, members, areas, member_types, title="Truss Structure", show_weights=True, note=None):
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

    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], color='k', s=10)
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    if note:
        ax.text2D(0.05, 0.95, note, transform=ax.transAxes, fontsize=12, color='red')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    coords, members, member_areas, moved_nodes = load_data()
    member_types = np.array([i % a for i in range(len(members))])
    coords[:37] = moved_nodes

    plot_truss(coords, members, member_areas, member_types, title="Initial Optimized Truss", show_weights=True)

    max_load = get_max_load_per_node(coords, members, member_areas, member_types)

    max_mass_per_node = max_load / g
    total_mass = max_mass_per_node * max_nodes_with_load

    print(f"\n✅ Max Load per Node = {max_load:.0f} N ≈ {max_mass_per_node:.2f} kg")
    print(f"✅ Total Max Load = {max_load * max_nodes_with_load:.0f} N ≈ {total_mass:.2f} kg")

    plot_truss(
        coords,
        members,
        member_areas,
        member_types,
        title="Max Load Configuration",
        show_weights=True,
        note=f"Max Load per Node = {max_load} N ≈ {max_mass_per_node:.2f} kg\n"
             f"Total = {max_load * max_nodes_with_load} N ≈ {total_mass:.2f} kg"
    )
