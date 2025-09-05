# ⚙️ 120-Bar Truss Optimization

This project tackles a well-known and challenging problem in **structural optimization**: the **120-bar truss**. The primary goal is to **minimize the total weight** of the structure while satisfying mechanical and geometric constraints.

P.S. OptiTruss is a little project of mine and I don't care what you guys will use it for and how you'll use it. So enjoy and have fun!
Sepehr Sohrabi

---

## 📝 Project Overview

- **Material:** Steel  
- **Design Variables:**  
  - Cross-sectional areas of each bar (0.0001 – 0.01293 m²)
  - Coordinates of intermediate nodes (highlighted with red circles in the reference figure)  
- **Objective Function:** Minimize the total weight of the structure  

---

## ⚡ Loading Scenarios

- **Scenario 1:**  
  - Node 1 → 3000 kg non-structural mass  
  - Nodes 2–13 → 1500 kg each  
  - Remaining nodes → 100 kg each  

- **Scenario 2 & 3:**  
  - 2 External Loads added  
  - Considers wind 

---

## 🛠 Implementation Tasks

1. **Cross-Sectional Assignment**  
   - The number of distinct bar types is determined by `(last digit of student ID + 10)`  
   - Each bar is assigned to one of these types  

2. **Optimization Using Evolutionary Algorithms**  
   - **Genetic Algorithm (GA)**  
   - **Particle Swarm Optimization (PSO)**  
   - Compare **convergence speed** and **solution accuracy**  

3. **Third Evolutionary Method**  
   - Apply a different algorithm (e.g., Differential Evolution, Cuckoo Search, etc.)  
   - Compare results with GA and PSO  

4. **Special Loading Analysis**  
   - Apply loads to nodes 1–37 (non-uniform)  
   - Determine the **maximum total load** without violating structural constraints  

5. **Bonus (Optional)**  
   - Repeat optimization with **dynamic constraints**:  
     - First natural frequency = 9 Hz  
     - Second natural frequency = 11 Hz  

---

## 📊 Outputs

- Optimized cross-sectional profiles of truss members  
- Optimized coordinates of intermediate nodes  
- Performance comparison of GA, PSO, and DE  
- Analysis of loading effects on the solution  
- Convergence plots and numerical results  
Update 
---

For additional information please contact me via E-mail: sepehrsohrabi47@gmail.com / sepehr_sohrabi@iust.ac.ir
