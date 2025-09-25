from __future__ import annotations

import streamlit as st


def apply_page_style() -> None:
    st.markdown(
        """
        <style>
        :root {
          --brand-primary: #4F46E5;
          --brand-secondary: #14B8A6;
          --brand-bg: #0B1021;
          --brand-card: #111735;
          --brand-text: #E5E7EB;
          --brand-muted: #9CA3AF;
        }
        .stApp {
          background: radial-gradient(1200px 600px at 10% 0%, rgba(79,70,229,0.25), transparent 50%),
                      radial-gradient(900px 500px at 90% 20%, rgba(20,184,166,0.25), transparent 50%),
                      var(--brand-bg);
          color: var(--brand-text);
        }
        .block-container { padding-top: 2rem; max-width: 1200px; }
        header[data-testid="stHeader"] { background: transparent; }
        section[data-testid="stSidebar"] {
          background: linear-gradient(180deg, rgba(17,23,53,0.95), rgba(11,16,33,0.95));
          border-right: 1px solid rgba(255,255,255,0.06);
        }
        .stButton>button, .stDownloadButton>button {
          background: linear-gradient(135deg, var(--brand-primary), var(--brand-secondary));
          color: white; border: 0; border-radius: 10px; padding: 0.6rem 1rem; font-weight: 600;
          box-shadow: 0 8px 20px rgba(79,70,229,0.35);
          transition: all 0.2s ease;
        }
        .stButton>button:hover, .stDownloadButton>button:hover { 
          filter: brightness(1.05); 
          transform: translateY(-1px);
          box-shadow: 0 12px 30px rgba(79,70,229,0.45);
        }
        .metric-card { background: var(--brand-card); border: 1px solid rgba(255,255,255,0.06); padding: 18px; border-radius: 14px; }
        .card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); padding: 18px; border-radius: 14px; box-shadow: 0 10px 30px rgba(0,0,0,0.25); }
        .muted { color: var(--brand-muted); }
        .title-gradient { background: linear-gradient(90deg, #fff, #c7d2fe 60%, #99f6e4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        /* Minimize the heavy uploader dropzone styling */
        div[data-testid="stFileUploader"] section { padding: 0 !important; }
        div[data-testid="stFileUploaderDropzone"] {
          background: transparent !important;
          border: 0 !important;
          padding: 0 !important;
          min-height: 0 !important;
          display: inline-block !important;
        }
        /* Hide helper text inside dropzone, keep only the button */
        div[data-testid="stFileUploaderDropzone"] > div:first-child { display: none !important; }
        /* Style the internal Browse button to match gradient buttons */
        div[data-testid="stFileUploaderDropzone"] button {
          background: linear-gradient(135deg, var(--brand-primary), var(--brand-secondary)) !important;
          color: #fff !important;
          border: 0 !important;
          border-radius: 10px !important;
          padding: 0.6rem 1.2rem !important;
          font-weight: 600 !important;
          box-shadow: 0 8px 20px rgba(79,70,229,0.35) !important;
          transition: all 0.2s ease !important;
          font-size: 14px !important;
        }
        div[data-testid="stFileUploaderDropzone"] button:hover { 
          filter: brightness(1.05) !important; 
          transform: translateY(-1px) !important;
          box-shadow: 0 12px 30px rgba(79,70,229,0.45) !important;
        }
        /* Style file uploader label */
        div[data-testid="stFileUploader"] label {
          font-weight: 500 !important;
          color: var(--brand-text) !important;
          margin-bottom: 0.5rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    left, right = st.columns([0.8, 0.2])
    with left:
        st.markdown("<h1 class='title-gradient'>ML Studio · Árboles y Ensambles</h1>", unsafe_allow_html=True)
        st.caption("Universidad Industrial de Santander · · Previo ML")
    with right:
        st.empty()


def render_footer() -> None:
    st.markdown(
        """
        <div class="muted" style="margin-top: 2rem; text-align: center;">
          UIS · 2025
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_branding(names: list[str]) -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 0.75rem 0 1.0rem 0;">
              <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Universidad_Industrial_de_Santander_logo.svg/2560px-Universidad_Industrial_de_Santander_logo.svg.png" alt="UIS" style="width:200px; filter: drop-shadow(0 2px 8px rgba(0,0,0,0.25));"/>
              <div style="margin-top: 0.5rem; font-weight: 700;">Universidad Industrial de Santander</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        if names:
            st.markdown(f"**{names[0]}** -- 2258059")
            st.markdown(f"**{names[1]}** - 2238039")


