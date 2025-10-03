import streamlit as st
from modules.ui import apply_page_style, render_header, render_footer, render_sidebar_branding
from modules.pages import page_data_eda, page_modeling, page_inference


def main() -> None:
    st.set_page_config(
        page_title="ML Studio · Árboles y Ensambles",
        page_icon="🌳",
        layout="wide",
        initial_sidebar_state="expanded",
    )  #  titulo de pestaña, expanded - apenas abrir pagina se muestre la parte izquierda

    apply_page_style()  # modulo ui
    render_header()       # modulo ui
    render_sidebar_branding(names=["Fabian Perez", "Nicolás Cabrera Ruiz"])        # modulo ui

    df_loaded = (
        st.session_state.get("file_processed", False) and 
        st.session_state.get("df") is not None and 
        not getattr(st.session_state.get("df"), "empty", True)
    )  #comprobar si se ha procesado un data set
    
    model_trained = st.session_state.get("trained_pipeline") is not None
    tabs = ["📊 Datos & EDA"] + (["🤖 Modelado"] if df_loaded else []) + (["🧪 Inferencia"] if model_trained else [])
    t = st.tabs(tabs) #secuencia logica de que las pestañas se van habilitando a medida que se haga una acción (subir archivo, entrenar)
    
    with t[0]:
        page_data_eda()
    
    if df_loaded:
        with t[1]:
            page_modeling()
    if model_trained:
        with t[2 if df_loaded else 1]:
            page_inference()

    render_footer() #modelo ui 


if __name__ == "__main__":
    main()


