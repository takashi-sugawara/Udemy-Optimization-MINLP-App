# 🚀 MINLP Master Visualizer

An advanced, interactive dashboard designed to explore and visualize **Mixed-Integer Non-Linear Programming (MINLP)** problems.

This application demonstrates the power of decomposition algorithms in solving complex optimization problems that involve both discrete decisions (Integers) and non-linear physical/economic constraints.

---

## 🧐 What is MINLP?
Mixed-Integer Non-Linear Programming is one of the most challenging classes of mathematical optimization. It combines:
- **Integer Programming (MIP):** Logical choices, equipment counts, or "Yes/No" decisions.
- **Non-Linear Programming (NLP):** Surface curves, chemical reactions, or law of diminishing returns.

Because of this dual nature, the feasible region is often non-convex and disconnected, requiring sophisticated orchestrators like **MindtPy** to coordinate specialized solvers.

---

## ✨ Key Features

### ⛰️ 3D Objective Surface
Visualize the "Landscape" of your optimization problem. The surface is colored based on feasibility, allowing you to see exactly where the boundaries (C1, C2, C3) intersect with the objective function.

### 📈 Sensitivity Trace (Gap Analysis)
Automatically sweeps parameters to show how the optimal solution shifts. 
- **Blue Line (Relaxed):** Shows how the problem behaves if integers were allowed to be continuous.
- **Orange Line (MINLP):** Shows the "jumps" and penalties caused by discrete integer constraints.

### 👨‍🏫 Strategy Comparison
Switch between different MindtPy decomposition strategies:
- **OA (Outer Approximation):** The standard stable approach using linearizations.
- **ECP (Extended Cutting Plane):** A cutting-plane approach for convex/non-convex problems.
- **FP (Feasibility Pump):** A heuristic strategy focused on finding a valid integer point quickly.

---

## 🛠️ Technology Stack
- **Dashboard:** [Streamlit](https://streamlit.io/)
- **Modeling:** [Pyomo](http://www.pyomo.org/) (Python Optimization Modeling Objects)
- **Orchestrator:** [MindtPy](https://pyomo.readthedocs.io/en/stable/explanation/solvers/mindtpy.html)
- **NLP Solver:** [Ipopt](https://coin-or.github.io/Ipopt/) (COIN-OR Project)
- **MIP Solver:** [GLPK](https://www.gnu.org/software/glpk/)
- **Visualization:** [Plotly](https://plotly.com/)

---

## 🚀 Getting Started

### Local Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/minlp-visualizer.git
   cd minlp-visualizer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## 📜 Acknowledgements
This project is powered by the **COIN-OR** ecosystem and the **Pyomo** community. Special thanks to the developers of MindtPy for making complex MINLP decomposition accessible to the Python community.
