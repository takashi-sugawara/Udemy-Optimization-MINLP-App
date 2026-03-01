import streamlit as st
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import io
import contextlib
import shutil
import os
import sys

# --- Robust Solver Path Detection (Inspired by Sec8) ---
def get_solver_path(solver_name):
    # 1. Check in PATH
    path = shutil.which(solver_name)
    if path: return path
    
    # 2. Check the directory of the current Python executable (Reliable for Conda)
    bin_dir = os.path.dirname(sys.executable)
    path_in_bin = os.path.join(bin_dir, solver_name)
    if os.path.exists(path_in_bin): return path_in_bin

    # 3. Check common Conda/Linux paths on Streamlit Cloud
    for p in [f"/home/adminuser/.conda/bin/{solver_name}", f"/opt/conda/bin/{solver_name}", f"/usr/bin/{solver_name}"]:
        if os.path.exists(p): return p
    return None

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
        background-color: rgba(255, 255, 255, 0.05);
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-left: 5px solid #1f77b4;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
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
def solve_minlp(c1, c2, c3, strategy='OA', integer_constrained=True):
    model = pyo.ConcreteModel()
    
    # Discovery of solver paths
    ipopt_path = get_solver_path('ipopt')
    glpk_path = get_solver_path('glpk')
    
    # Force system path update for sub-solvers
    for p in [ipopt_path, glpk_path]:
        if p:
            dir_path = os.path.dirname(p)
            if dir_path not in os.environ['PATH']:
                os.environ['PATH'] = dir_path + os.pathsep + os.environ['PATH']

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
                # Specify executables directly if found, or fall back to PATH
                opt = SolverFactory('mindtpy')
                solve_kwargs = {
                    'mip_solver': 'glpk',
                    'nlp_solver': 'ipopt',
                    'strategy': strategy,
                    'tee': True
                }
                # If paths were found, MindtPy usually finds them in PATH if we updated os.environ
                opt.solve(model, **solve_kwargs)
            else:
                if ipopt_path:
                    opt = SolverFactory('ipopt', executable=ipopt_path)
                else:
                    opt = SolverFactory('ipopt')
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
                'm': m_main,
                'l': l_main,
                'rel': m_rel if s_rel is True else None
            }
            st.toast("Success!", icon="✅")
        else:
            st.error(f"Optimization Failed: {s_main}")
            if l_main:
                with st.expander("View Error Details"):
                    st.code(l_main)

# --- Visualization Render ---
if st.session_state.res:
    m = st.session_state.res['m']
    l = st.session_state.res['l']
    m_rel = st.session_state.res['rel']

    with tab_vis:
        # 1. Summary Metrics at the top
        st.markdown('<p class="metric-label">Optimal Objective Value</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">{pyo.value(m.obj):.4f}</p>', unsafe_allow_html=True)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.write(f"**Optimal x (Integer):** `{pyo.value(m.x):.2f}`")
            st.write(f"**Optimal y (Continuous):** `{pyo.value(m.y):.4f}`")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_m2:
            if m_rel:
                st.info(f"💡 Relaxation GAP: **{((pyo.value(m_rel.obj) - pyo.value(m.obj))/pyo.value(m_rel.obj)*100):.2f}%**")

        st.divider()

        # 2. 2D Visualization (Contour + Constraints) 
        st.subheader("🗺️ 2D Feasibility & Contour Map")
        
        # Grid for contour
        x_2d = np.linspace(0, 10, 100)
        y_2d = np.linspace(0, 10, 100)
        X2, Y2 = np.meshgrid(x_2d, y_2d)
        Z2 = X2 + Y2 * X2

        fig2 = go.Figure()
        
        # Contour
        fig2.add_trace(go.Contour(
            z=Z2, x=x_2d, y=y_2d,
            colorscale='Viridis',
            contours_coloring='heatmap',
            name='Objective',
            opacity=0.8
        ))

        # Constraints
        # C1: -x + 2yx = c1 => y = (c1 + x) / 2x
        y_c1 = (c1_val + x_2d[1:]) / (2 * x_2d[1:])
        fig2.add_trace(go.Scatter(x=x_2d[1:], y=y_c1, name='C1 Boundary', line=dict(color='red', dash='dash')))
        
        # C2: 2x + y = c2 => y = c2 - 2x
        y_c2 = c2_val - 2 * x_2d
        fig2.add_trace(go.Scatter(x=x_2d, y=y_c2, name='C2 Boundary', line=dict(color='yellow', dash='dash')))
        
        # C3: 2x - y = c3 => y = 2x - c3
        y_c3 = 2 * x_2d - c3_val
        fig2.add_trace(go.Scatter(x=x_2d, y=y_c3, name='C3 Boundary', line=dict(color='orange', dash='dash')))

        # Optimal Point
        fig2.add_trace(go.Scatter(
            x=[pyo.value(m.x)], y=[pyo.value(m.y)],
            mode='markers+text',
            marker=dict(size=15, color='orange', symbol='diamond', line=dict(color='white', width=2)),
            name='Integer Optimal',
            text=["Optimal"], textposition="top center"
        ))

        fig2.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title='x',
            yaxis_title='y',
            yaxis=dict(range=[0, 10]),
            xaxis=dict(range=[0, 10]),
            height=600,
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.info("The dashed lines represent constraints. The optimal solution must stay within the lower intersection area.")

        st.divider()

        # 3. 3D Visualization
        st.subheader("🧊 3D Objective Optimization Surface")
        x_vals = np.linspace(0, 10, 50)
        y_vals = np.linspace(0, 10, 50)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = X + Y * X  # Objective function
        
        fig = go.Figure()
        fig.add_trace(go.Surface(
            x=x_vals, y=y_vals, z=Z,
            colorscale='Viridis',
            opacity=0.7,
            showscale=False,
            name='Objective Surface'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[pyo.value(m.x)], y=[pyo.value(m.y)], z=[pyo.value(m.obj)],
            mode='markers',
            marker=dict(size=10, color='orange', symbol='diamond', line=dict(color='white', width=2)),
            name='MINLP Optimal'
        ))
        
        if m_rel:
            fig.add_trace(go.Scatter3d(
                x=[pyo.value(m_rel.x)], y=[pyo.value(m_rel.y)], z=[pyo.value(m_rel.obj)],
                mode='markers',
                marker=dict(size=8, color='cyan', symbol='circle', line=dict(color='white', width=1)),
                name='NLP Relaxed'
            ))

        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            scene=dict(
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)', backgroundcolor='rgba(0,0,0,0)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)', backgroundcolor='rgba(0,0,0,0)'),
                zaxis=dict(gridcolor='rgba(255,255,255,0.1)', backgroundcolor='rgba(0,0,0,0)'),
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='Objective'
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=700
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("✨ Tip: Rotate the 3D plot to see how the optimal point sits on the surface gradient.")


        # Logs
        if show_logs:
            with st.expander("📋 MindtPy Solver Execution Logs", expanded=False):
                st.code(l)

    with tab_sens:
        st.subheader("📈 Sensitivity Study: Constraint Boundaries")
        st.write("Visualizing how varying $c_1$ (Constraint 1 constant) impacts the final outcome.")
        
        c1_range = np.linspace(max(0, c1_val-5), c1_val+5, 10)
        sens_data = []
        
        with st.status("Walking through parameter space...") as status:
            for test_c in c1_range:
                # Quick solve for sensitivity (silent)
                model_s, ok, _ = solve_minlp(test_c, c2_val, c3_val, strategy='OA')
                if ok:
                    sens_data.append({
                        "Constraint C1": test_c,
                        "Objective": pyo.value(model_s.obj),
                        "Type": "MINLP (Integer)"
                    })
                
                # Also solve relaxed for comparison
                model_r, ok_r, _ = solve_minlp(test_c, c2_val, c3_val, strategy='OA', integer_constrained=False)
                if ok_r:
                    sens_data.append({
                        "Constraint C1": test_c,
                        "Objective": pyo.value(model_r.obj),
                        "Type": "Relaxed (NLP)"
                    })
            status.update(label="Sensitivity Map Generated!", state="complete")

        df_sens = pd.DataFrame(sens_data)
        fig_sens = px.line(df_sens, x="Constraint C1", y="Objective", color="Type", 
                           markers=True,
                           color_discrete_map={"MINLP (Integer)": "orange", "Relaxed (NLP)": "cyan"},
                           title="Impact of Constraint C1 on Global Utility")
        
        fig_sens.add_vline(x=c1_val, line_dash="dash", line_color="red", annotation_text="Current C1")
        st.plotly_chart(fig_sens, use_container_width=True)

    with tab_math:
        st.subheader("📝 Mathematical Definition")
        st.latex(r" \max_{x, y} \quad Z = x + yx ")
        st.latex(r" \text{subject to:} ")
        st.latex(rf" -x + 2yx \le {c1_val:.2f} \quad \text{{(Non-linear Constraint)}} ")
        st.latex(rf" 2x + y \le {c2_val:.2f} ")
        st.latex(rf" 2x - y \le {c3_val:.2f} ")
        st.latex(r" x \in \{0, 1, \dots, 10\} \quad \text{(Integer constraint)} ")
        st.latex(r" y \in [0, 10] \quad \text{(Continuous constraint)} ")
        
        st.success("This model is a classic Mixed-Integer Non-Linear Programming (MINLP) problem because it contains both integer variables and non-linear terms ($yx$).")
else:
    with tab_vis:
        st.info("Adjust the parameters in the sidebar and click **RUN OPTIMIZATION** to begin.")
