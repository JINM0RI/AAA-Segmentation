import streamlit as st
import os
import sys
import shutil
import tempfile
import time
from pathlib import Path
from PIL import Image
import numpy as np

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AortaSeg · Segmentation Tool",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

  /* ── Root & Body ── */
  :root {
    --bg:        #0a0c10;
    --surface:   #111318;
    --border:    #1e2330;
    --accent:    #00e5ff;
    --accent2:   #7b61ff;
    --text:      #dde3f0;
    --muted:     #5a6278;
    --danger:    #ff4d6d;
    --success:   #00e5a0;
  }

  html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
  }

  [data-testid="stHeader"] { background: transparent !important; }
  [data-testid="stToolbar"] { display: none; }
  footer { visibility: hidden; }

  /* ── Hide sidebar toggle ── */
  [data-testid="collapsedControl"] { display: none; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  /* ── Main container ── */
  .main .block-container {
    max-width: 1100px;
    padding: 2.5rem 2rem 4rem;
    margin: 0 auto;
  }

  /* ── Header ── */
  .app-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.5rem;
  }

  .app-logo {
    width: 42px;
    height: 42px;
    border-radius: 10px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    flex-shrink: 0;
  }

  .app-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.7rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1 !important;
  }

  .app-subtitle {
    color: var(--muted);
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.2rem;
  }

  .divider {
    height: 1px;
    background: var(--border);
    margin: 1.5rem 0 2rem;
  }

  /* ── Upload Zone ── */
  [data-testid="stFileUploader"] > div:first-child {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 14px !important;
    padding: 2.5rem 2rem !important;
    transition: border-color 0.2s, background 0.2s !important;
  }

  [data-testid="stFileUploader"] > div:first-child:hover {
    border-color: var(--accent) !important;
    background: #111c24 !important;
  }

  [data-testid="stFileUploader"] label {
    color: var(--muted) !important;
    font-family: 'DM Mono', monospace !important;
  }

  /* ── Buttons ── */
  [data-testid="baseButton-secondary"],
  [data-testid="baseButton-primary"],
  .stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.8rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.1s !important;
  }

  .stButton > button:hover {
    opacity: 0.85 !important;
    transform: translateY(-1px) !important;
  }

  .stButton > button:disabled {
    background: var(--border) !important;
    color: var(--muted) !important;
    cursor: not-allowed !important;
    transform: none !important;
  }

  /* ── Progress bar ── */
  [data-testid="stProgressBar"] > div {
    background: var(--border) !important;
    border-radius: 6px !important;
    overflow: hidden;
  }

  [data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
    border-radius: 6px !important;
    transition: width 0.3s ease !important;
  }

  /* ── Result card ── */
  .result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
  }

  .result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
  }

  .result-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.75rem;
  }

  /* ── Status pill ── */
  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #0d1f17;
    border: 1px solid var(--success);
    color: var(--success);
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    margin-bottom: 1.2rem;
  }

  .status-dot {
    width: 7px;
    height: 7px;
    background: var(--success);
    border-radius: 50%;
    animation: pulse 1.5s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
  }

  /* ── Step labels ── */
  .step-log {
    background: #0a0c10;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.75rem;
    color: var(--muted);
    line-height: 1.9;
    font-family: 'DM Mono', monospace;
  }

  .step-log .done  { color: var(--success); }
  .step-log .active { color: var(--accent); }
  .step-log .wait  { color: var(--muted); }

  /* ── Image containers ── */
  [data-testid="stImage"] img {
    border-radius: 10px !important;
    width: 100% !important;
  }

  /* ── Metric tags ── */
  .meta-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-top: 1rem;
  }

  .meta-tag {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.25rem 0.7rem;
    font-size: 0.7rem;
    color: var(--muted);
  }

  .meta-tag span { color: var(--text); font-weight: 500; }

  /* ── Warning / error ── */
  .warn-box {
    background: #1a0c10;
    border: 1px solid var(--danger);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    color: var(--danger);
    font-size: 0.8rem;
  }

  /* ── Section header ── */
  .section-head {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.8rem;
  }

  /* Hide streamlit elements */
  #MainMenu { visibility: hidden; }
  [data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_PATH  = "outputs/models/best_model.pth"
THRESHOLD   = 0.95
KERNEL      = (5, 5)
VIS_DIR     = "outputs/marked_mask"
OVERLAY_DIR = "outputs/marked_mask/overlay"
TEST_DIR    = "data/test"

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-logo">🫀</div>
  <div>
    <div class="app-title">Hybrid Attention-U
Net and Swin Transformer Model</div>
    <div class="app-subtitle">Abdominal Aorta Segmentation · Deep Learning</div>
  </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ─── Model check ──────────────────────────────────────────────────────────────
model_exists = os.path.isfile(MODEL_PATH)

if not model_exists:
    st.markdown(f"""
    <div class="warn-box">
      ⚠️ Model not found at <code>{MODEL_PATH}</code>.<br>
      Please ensure the model file exists before running inference.
    </div>
    """, unsafe_allow_html=True)

# ─── Layout: two columns ──────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-head">01 — Upload Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        label="Drop a DICOM, PNG, or JPG image here",
        type=["png", "jpg", "jpeg", "dcm", "tiff", "tif", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded:
        try:
            img = Image.open(uploaded)
            st.markdown('<div class="result-card"><div class="result-label">Input Preview</div>', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            w, h = img.size
            st.markdown(f"""
            <div class="meta-row">
              <div class="meta-tag">Size <span>{w} × {h} px</span></div>
              <div class="meta-tag">Mode <span>{img.mode}</span></div>
              <div class="meta-tag">File <span>{uploaded.name}</span></div>
            </div>
            </div>""", unsafe_allow_html=True)
        except Exception:
            st.markdown('<div class="result-card"><div class="result-label">Input Preview</div>', unsafe_allow_html=True)
            st.info("Preview unavailable for this file type.")
            st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-head">02 — Segmentation Output</div>', unsafe_allow_html=True)

    run_btn = st.button(
        "▶ Run Inference",
        disabled=(not uploaded or not model_exists),
        use_container_width=True,
    )

    result_placeholder = st.empty()

    if not uploaded:
        result_placeholder.markdown("""
        <div class="result-card" style="min-height:220px;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:0.8rem;">
          Upload an image to begin
        </div>
        """, unsafe_allow_html=True)

    if run_btn and uploaded and model_exists:
        # ── Setup temp dirs ───────────────────────────────────────────────────
        with tempfile.TemporaryDirectory() as tmp_root:
            tmp_input  = os.path.join(tmp_root, "test")
            tmp_vis    = os.path.join(tmp_root, "vis")
            os.makedirs(tmp_input,  exist_ok=True)
            os.makedirs(tmp_vis,    exist_ok=True)

            # Save uploaded file
            in_path = os.path.join(tmp_input, uploaded.name)
            with open(in_path, "wb") as f:
                f.write(uploaded.getvalue())

            # ── Progress UI ───────────────────────────────────────────────────
            prog_bar  = st.progress(0)
            step_area = st.empty()

            STEPS = [
                ("Initialising model…",          12),
                ("Loading weights…",             28),
                ("Pre-processing image…",        46),
                ("Running forward pass…",        68),
                ("Applying threshold (0.95)…",   82),
                ("Morphological filtering…",     92),
                ("Saving mask & visualisation…", 100),
            ]

            def render_steps(done_count, active_idx):
                lines = []
                for i, (label, _) in enumerate(STEPS):
                    if i < done_count:
                        lines.append(f'<span class="done">✔ {label}</span>')
                    elif i == active_idx:
                        lines.append(f'<span class="active">◉ {label}</span>')
                    else:
                        lines.append(f'<span class="wait">○ {label}</span>')
                step_area.markdown(
                    '<div class="step-log">' + "<br>".join(lines) + "</div>",
                    unsafe_allow_html=True,
                )

            # Animate steps
            render_steps(0, 0)
            prog_bar.progress(0)

            try:
                # -- Step 0-1: model init (real import happens here) -----------
                time.sleep(0.4)
                render_steps(1, 1)
                prog_bar.progress(12)

                from inference.predict import run_inference  # noqa: E402

                time.sleep(0.3)
                render_steps(2, 2)
                prog_bar.progress(28)

                # -- Steps 2-5: animate while inference runs ------------------
                # We'll run inference in this thread and update UI before/after
                for step_idx in range(2, 5):
                    _, pct = STEPS[step_idx]
                    render_steps(step_idx, step_idx)
                    prog_bar.progress(pct)
                    time.sleep(0.35)

                # -- Actual inference -----------------------------------------
                render_steps(5, 5)
                prog_bar.progress(82)

                run_inference(
                    input_path=tmp_input,
                    model_path=MODEL_PATH,
                    threshold=THRESHOLD,
                    kernel_size=KERNEL,
                    save_visualizations=True,
                    vis_dir=tmp_vis,
                )

                render_steps(6, 6)
                prog_bar.progress(92)
                time.sleep(0.3)

                render_steps(7, 7)
                prog_bar.progress(100)
                time.sleep(0.25)

                # ── Find output (model writes to overlay/ subfolder) ──────────
                overlay_dir = Path(tmp_vis) / "overlay"
                search_root = overlay_dir if overlay_dir.exists() else Path(tmp_vis)
                out_files = []
                for ext in ("*.png", "*.jpg", "*.jpeg", "*.tiff", "*.bmp"):
                    out_files.extend(search_root.rglob(ext))

                prog_bar.empty()
                step_area.empty()

                if out_files:
                    mask_path = str(out_files[0])
                    mask_img  = Image.open(mask_path)

                    result_placeholder.markdown("""
                    <div class="status-pill">
                      <div class="status-dot"></div>Segmentation complete
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown('<div class="result-card"><div class="result-label">Predicted Mask</div>', unsafe_allow_html=True)
                    st.image(mask_img, use_container_width=True)

                    mw, mh = mask_img.size
                    arr = np.array(mask_img)
                    pos_px  = int((arr > 127).sum()) if arr.ndim == 2 else int((arr.mean(-1) > 127).sum())
                    total   = mw * mh
                    coverage = f"{100 * pos_px / total:.1f}%"

                    st.markdown(f"""
                    <div class="meta-row">
                      <div class="meta-tag">Output <span>{mw} × {mh} px</span></div>
                      <div class="meta-tag">Threshold <span>{THRESHOLD}</span></div>
                      <div class="meta-tag">Aorta coverage <span>{coverage}</span></div>
                    </div>
                    </div>""", unsafe_allow_html=True)

                    # Download button
                    with open(mask_path, "rb") as f:
                        st.download_button(
                            label="⬇ Download Mask",
                            data=f.read(),
                            file_name=f"mask_{uploaded.name}",
                            mime="image/png",
                            use_container_width=True,
                        )
                else:
                    prog_bar.empty()
                    step_area.empty()
                    st.markdown("""
                    <div class="warn-box">
                      ⚠️ Inference ran but no output file was found in the visualisation directory.
                      Check that <code>save_visualizations=True</code> writes to the expected path.
                    </div>
                    """, unsafe_allow_html=True)

            except ImportError as e:
                prog_bar.empty()
                step_area.empty()
                st.markdown(f"""
                <div class="warn-box">
                  ⚠️ Could not import <code>inference.predict</code>.<br>
                  Make sure the module is in your Python path and dependencies are installed.<br><br>
                  <code>{e}</code>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                prog_bar.empty()
                step_area.empty()
                st.markdown(f"""
                <div class="warn-box">
                  ⚠️ Inference failed with an unexpected error:<br><br>
                  <code>{e}</code>
                </div>
                """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;text-align:center;color:var(--muted);font-size:0.68rem;letter-spacing:0.1em;">
  AORTASEG · ABDOMINAL AORTA SEGMENTATION · MODEL: best_model.pth · THRESHOLD 0.95
</div>
""", unsafe_allow_html=True)