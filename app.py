import streamlit as st
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from amplpy import modules
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import io
import contextlib
import shutil

# --- Universal Solver Setup ---
# This ensures solvers can be found on both Streamlit Cloud (Auto-install) and Local Mac
def get_solver_path(solver_name):
    # Try system PATH
    path = shutil.which(solver_name)
    if path:
        return path
    
    # Try amplpy-modules discovery
    try:
        from amplpy import modules
        return modules.find(solver_name)
    except:
        return None

def check_and_install_solvers():
    # Determine which solvers are missing
    missing = []
    if get_solver_path('ipopt') is None:
        missing.append('coin')  # Ipopt lives in the 'coin' module
    if get_solver_path('glpk') is None:
        missing.append('glpk')

    if missing:
        # Remove duplicates just in case
        missing = list(set(missing))
        with st.status(f"🛠️ Installing missing solvers: {', '.join(missing)}...") as status:
            try:
                from amplpy import modules
                modules.install(missing)
                # After installation, refresh PATH by re‑adding the directories where the binaries were placed
                for solver in ['ipopt', 'glpk']:
                    p = modules.find(solver)
                    if p:
                        dir_path = os.path.dirname(p)
                        if dir_path not in os.environ.get('PATH', ''):
                            os.environ['PATH'] = f"{dir_path}:{os.environ.get('PATH', '')}"
                status.update(label="✅ Solvers installed successfully!", state="complete", expanded=False)
                # Force a full rerun so the newly‑installed executables are discovered
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to install solvers: {e}")
                status.update(label="❌ Installation failed", state="error")

# Page Config
st.set_page_config(
    page_title="MINLP Master Visualizer",
    page_icon="🚀",
    layout="wide",
)

# --- Premium UI Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main {
        background-color: #f0f2f6;
    }
    .metric-card {
        background-color: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 5px solid #1f77b4;
        margin-bottom: 20px;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🚀 MINLP Master Visualizer")
st.caption("Mixed-Integer Non-Linear Programming Dashboard | Powered by MindtPy & COIN-OR")

# Check and install solvers if missing (one-time on app start)
check_and_install_solvers()

# --- Sidebar: Expert Controls ---
with st.sidebar:
    st.header("⚙️ Solver Configuration")
    
    with st.expander("🛠️ Constraints (c1, c2, c3)", expanded=True):
        c1_val = st.slider("C1 constant (-x + 2yx ≤ ...)", 0.0, 20.0, 8.0, 0.5)
        c2_val = st.slider("C2 constant (2x + y ≤ ...)", 5.0, 25.0, 14.0, 0.5)
        c3_val = st.slider("C3 constant (2x - y ≤ ...)", 5.0, 25.0, 10.0, 0.5)

    with st.expander("👨‍🏫 Strategy & Mode", expanded=True):
        strategy = st.selectbox("MindtPy Strategy", 
                                ["OA", "ECP", "FP"], 
                                index=0,
                                help="OA: Outer Approximation (Stable), ECP: Extended Cutting Plane, FP: Feasibility Pump (Alternative)")
        show_relaxation = st.toggle("Compare with Relaxation (NLP)", value=True)
        show_logs = st.toggle("Show Live Progress Logs", value=True)

    st.divider()
    run_button = st.button("RUN OPTIMIZATION", type="primary", use_container_width=True)
    
    st.divider()
    st.markdown("### 🧬 Architecture")
    st.info(f"Orchestrator: **MindtPy ({strategy})**\n\nDiscrete Solver: **GLPK**\n\nContinuous Solver: **Ipopt**")

# --- Optimization Engine ---
import os

def solve_minlp(c1, c2, c3, strategy='OA', integer_constrained=True):
    model = pyo.ConcreteModel()
    
    # Discovery of solver paths and add to PATH for sub-solvers
    ipopt_path = get_solver_path('ipopt')
    glpk_path = get_solver_path('glpk')
    
    # Add solver directories to system PATH so MindtPy sub-solvers can find them
    new_paths = []
    for p in [ipopt_path, glpk_path]:
        if p:
            dir_path = os.path.dirname(p)
            if dir_path not in os.environ['PATH']:
                new_paths.append(dir_path)
    
    if new_paths:
        os.environ['PATH'] = os.pathsep.join(new_paths) + os.pathsep + os.environ['PATH']

    domain = pyo.Integers if integer_constrained else pyo.Reals
    model.x = pyo.Var(within=domain, bounds=(0, 10))
    model.y = pyo.Var(bounds=(0, 10))
    
    model.C1 = pyo.Constraint(expr= -model.x + 2 * model.y * model.x <= c1)
    model.C2 = pyo.Constraint(expr= 2 * model.x + model.y <= c2)
    model.C3 = pyo.Constraint(expr= 2 * model.x - model.y <= c3)
    model.obj = pyo.Objective(expr= model.x + model.y * model.x, sense=pyo.maximize)
    
    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
        try:
            if integer_constrained:
                # Use MindtPy for Mixed-Integer
                opt = SolverFactory('mindtpy')
                opt.solve(model, 
                          mip_solver='glpk', 
                          nlp_solver='ipopt', 
                          strategy=strategy, 
                          tee=True)
            else:
                # Use simple Ipopt for Relaxation (NLP)
                opt = SolverFactory('ipopt') # Should be in PATH now
                opt.solve(model, tee=True)
            return model, True, log_buffer.getvalue()
        except Exception as e:
            return None, str(e), log_buffer.getvalue()

# Session State
if 'res' not in st.session_state:
    st.session_state.res = None

# --- Main App Logic ---
tab_vis, tab_sens, tab_math = st.tabs(["📊 ANALYSIS & VISUALIZATION", "📈 SENSITIVITY TRACE", "📝 MATH FORMULATION"])

if run_button:
    with st.spinner(f"Orchestrating {strategy} strategy..."):
        m_main, s_main, l_main = solve_minlp(c1_val, c2_val, c3_val, strategy=strategy)
        m_rel, s_rel, _ = solve_minlp(c1_val, c2_val, c3_val, integer_constrained=False) if show_relaxation else (None, False, "")
        
        # Save results
        if s_main is True:
            st.session_state.res = {
                "x": pyo.value(m_main.x), "y": pyo.value(m_main.y), "obj": pyo.value(m_main.obj),
                "x_rel": pyo.value(m_rel.x) if s_rel is True else None,
                "y_rel": pyo.value(m_rel.y) if s_rel is True else None,
                "obj_rel": pyo.value(m_rel.obj) if s_rel is True else None,
                "logs": l_main, "status": "Optimal"
            }
        else:
            st.session_state.res = {"status": "Infeasible", "error": s_main, "logs": l_main}

with tab_math:
    st.header("📝 Mathematical Model")
    st.latex(r"\max \quad Z = x + y \cdot x")
    st.latex(rf"s.t. \quad -x + 2yx \le {c1_val} \quad (C1)")
    st.latex(rf" \quad \quad 2x + y \le {c2_val} \quad (C2)")
    st.latex(rf" \quad \quad 2x - y \le {c3_val} \quad (C3)")
    st.latex(r"x \in \mathbb{Z} \cap [0, 10], \quad y \in \mathbb{R} \cap [0, 10]")

with tab_vis:
    if st.session_state.res:
        res = st.session_state.res
        
        if res["status"] == "Infeasible":
            error_msg = str(res.get('error', 'Unknown Error'))
            if "dlsym" in error_msg or "symbol not found" in error_msg:
                st.error("⚙️ **SYSTEM / SOLVER ERROR**")
                st.warning(f"Technical details: {error_msg}")
                st.info("This usually means the selected strategy requires system libraries or solvers that are not fully compatible with your current OS/Build. Please try switching back to **OA** strategy.")
            else:
                st.error(f"❌ **MATHEMATICAL INFEASIBILITY**")
                st.warning(f"The solver reported: {error_msg}")
                st.info("Try loosening the constraints (increasing c1, c2, or c3). The feasible region might be empty.")
        else:
            # Metric Cards
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">Optimal X</div><div class="metric-value">{int(round(res["x"]))}</div></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">Optimal Y</div><div class="metric-value">{res["y"]:.3f}</div></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="metric-card"><div class="metric-label">Objective Z</div><div class="metric-value">{res["obj"]:.3f}</div></div>', unsafe_allow_html=True)
            with c4:
                gap = abs(res["obj"] - res["obj_rel"]) if res["obj_rel"] else 0
                st.markdown(f'<div class="metric-card" style="border-left-color: #ff7f0e;"><div class="metric-label">Integrality Gap</div><div class="metric-value">{gap:.3f}</div></div>', unsafe_allow_html=True)

            if show_logs:
                with st.expander("📂 Iteration Logs & Solver Strategy Trace", expanded=False):
                    st.code(res["logs"])

            # Visualization
            x_g = np.linspace(0, 10, 100)
            y_g = np.linspace(0, 10, 100)
            X, Y = np.meshgrid(x_g, y_g)
            Z = X + Y * X
            feas = ((-X + 2*Y*X <= c1_val) & (2*X + Y <= c2_val) & (2*X - Y <= c3_val)).astype(float)
            
            # --- 2D Plot Section ---
            st.subheader("📍 2D Feasibility & Optimal Point")
            fig2d = go.Figure()
            fig2d.add_trace(go.Contour(x=x_g, y=y_g, z=feas, showscale=False, colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0, 255, 0, 0.2)']], name="Feasible"))
            fig2d.add_trace(go.Contour(x=x_g, y=y_g, z=Z, colorscale='Blues', opacity=0.3, name="Objective"))
            fig2d.add_trace(go.Scatter(x=[res['x']], y=[res['y']], mode='markers+text', marker=dict(size=15, color='orange', symbol='star'), text=["MINLP ★"], name="MINLP"))
            if res['x_rel']:
                fig2d.add_trace(go.Scatter(x=[res['x_rel']], y=[res['y_rel']], mode='markers', marker=dict(size=10, color='blue'), name="Relaxed"))
            
            fig2d.update_layout(xaxis_title="X", yaxis_title="Y", height=600, margin=dict(l=0,r=0,b=0,t=30), template="plotly_white")
            st.plotly_chart(fig2d, use_container_width=True)

            st.divider()

            # --- 3D Plot Section ---
            st.subheader("⛰️ 3D Objective Surface")
            st.info("💡 **Interactive Tip**: Use your mouse to rotate (drag), zoom (scroll), and pan (right-click drag) the 3D surface below!")
            
            Z_masked = np.where(feas > 0, Z, np.nan)
            fig3d = go.Figure(data=[go.Surface(x=x_g, y=y_g, z=Z, opacity=0.2, colorscale='Greys', showscale=False, hoverinfo='skip')])
            fig3d.add_trace(go.Surface(x=x_g, y=y_g, z=Z_masked, colorscale='Viridis', name="Feasible Surface", colorbar=dict(title="Z Value", x=1.1)))
            fig3d.add_trace(go.Scatter3d(x=[res['x']], y=[res['y']], z=[res['obj']], mode='markers', marker=dict(size=8, color='orange', symbol='diamond'), name="Optimal Solution"))
            
            fig3d.update_layout(
                scene=dict(
                    xaxis_title='X', yaxis_title='Y', zaxis_title='Z (Objective)',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)) # Better initial angle
                ),
                height=700,
                margin=dict(l=0,r=0,b=0,t=30)
            )
            st.plotly_chart(fig3d, use_container_width=True)
    else:
        st.info("Click 'RUN OPTIMIZATION' to start the analysis.")

with tab_sens:
    st.header("📈 Sensitivity Trace: Impact of C1")
    st.write("We sweep the C1 value from 0 to 20 to see how it shifts the optimal objective.")
    if st.button("Generate Sensitivity Analysis"):
        with st.spinner("Sweeping C1 values..."):
            trace_data = []
            # Performance note: Increased points for smoother curves
            for test_c1 in np.linspace(0, 20, 15):
                # Run MINLP
                m, s, _ = solve_minlp(test_c1, c2_val, c3_val, integer_constrained=True)
                if s is True: trace_data.append({"C1": test_c1, "Obj": pyo.value(m.obj), "Type": "MINLP"})
                
                # Run Relaxed
                m2, s2, _ = solve_minlp(test_c1, c2_val, c3_val, integer_constrained=False)
                if s2 is True: trace_data.append({"C1": test_c1, "Obj": pyo.value(m2.obj), "Type": "Relaxed"})
            
            df = pd.DataFrame(trace_data)
            fig_sens = px.line(df, x="C1", y="Obj", color="Type", markers=True, 
                               title="Objective Value vs C1 Constant Sensitivity",
                               color_discrete_map={"MINLP": "#ff7f0e", "Relaxed": "#1f77b4"})
            
            fig_sens.update_layout(template="plotly_white", height=500)
            st.plotly_chart(fig_sens, use_container_width=True)
            st.markdown("""
            **What we learn:**
            - The **Blue Line (Relaxed)** is smooth because it's a continuous problem.
            - The **Orange Line (MINLP)** often shows 'jumps'. This happens when the optimal integer $x$ suddenly shifts from one value to another (e.g., from 5 to 6) as the constraint is loosened.
            """)
