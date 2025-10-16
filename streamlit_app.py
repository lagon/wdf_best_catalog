import streamlit as st


st.set_page_config(
    page_title="BEST AI Search Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stMarkdownContainer"] p {
        font-size: 1.5rem;
        margin: 1rem;
        width: 500px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container(horizontal_alignment="center", vertical_alignment="center"):
    if st.button("Hierarchical search"):
        st.switch_page("pages/1_hierarchical.py")
    if st.button("Complete search"):
        st.switch_page("pages/2_complete.py")
    if st.button("Batched hierarchical"):
        st.switch_page("pages/3_batched_hierarchy.py")
    if st.button("Flat search __(new)__"):
        st.switch_page("pages/4_flat_search.py")
    if st.button("Chatbot Search __(new)__"):
        st.switch_page("pages/5_chatbot_search.py")
