import os

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
import streamlit as st

from catalog import Catalog

INPUT_HEIGHT = 100
OPENAI_MODEL = "gpt-4.1"
VDB_PATH = "catalog_db"

client = OpenAI()
catalog = Catalog(VDB_PATH)

st.set_page_config(
    page_title="BEST AI Search Engine",
    initial_sidebar_state="expanded",
    layout="wide",
)
st.markdown(
    """
    <style>
    [data-baseweb="tab-list"] [data-testid="stMarkdownContainer"] {
        font-size: 1.2rem;
        padding: 0.5rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def clear_state():
    keys_to_clear = [
        "query_text",
        "query_embed",
        "category",
        "prod_family",
        "prod_group",
        "product",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


page = "batched_hierarchy"
if "page" in st.session_state and st.session_state["page"] != page:
    clear_state()
st.session_state["page"] = page


def preprocess_query(query: str) -> str:
    query.strip().replace("\n", ", ")
    prompt = f"""
    Rewrite the following product description into clear natural language for embedding.
    Expand abbreviations. Keep only essential product attributes like material, type,
    dimensions, shape, color, surface, layer details, fractions.
    Make it concise and unambiguous, but keep all product details.
    Remove unnecessary formatting or jargon.
    Input: "{query}"
    Output:
    """
    resp = client.responses.create(model=OPENAI_MODEL, input=prompt)
    output = resp.output_text.strip().replace("\n", " ")
    text = f"Original: {query}\nProcessed: {output}"
    return text


def is_match(requested_desc: str, product_desc: str) -> bool:
    prompt = f"""
    Compare the two product descriptions. If the catalog item can satisfy the
    requirements of the requested product, return YES. Otherwise, return NO.
    Requested product: {requested_desc}
    Catalog item: {product_desc}
    """
    resp = client.responses.create(model=OPENAI_MODEL, input=prompt)
    return "YES" in resp.output_text.upper()


def query_component() -> tuple[str, np.ndarray, bool] | None:
    c1, c2 = st.columns([1, 1])
    query_text = c1.text_area(
        label="query",
        label_visibility="collapsed",
        placeholder="Enter query here...",
        height=INPUT_HEIGHT,
    )
    to_preprocess_query = st.checkbox("LLM query preprocessing (todo)", disabled=True)
    to_verify_results = st.checkbox("LLM result verification (todo)", disabled=True)
    submitted = st.form_submit_button("Submit", type="primary")
    if not submitted or query_text == "":
        return None
    clear_state()
    if to_preprocess_query:
        with st.spinner("Processing query...", show_time=True):
            query_text = preprocess_query(query_text)
    query_embed = catalog.embed_document(query_text)
    with c2.container(border=True, height=INPUT_HEIGHT):
        st.markdown(query_text)
    return query_text, query_embed, to_verify_results


def kde_analysis(data) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.array(data)
    x_grid = np.linspace(data.min(), data.max(), 1000)
    if len(data) < 2:
        data = np.concatenate([data, data + 0.1], axis=0)
    density = gaussian_kde(data, bw_method=0.3)(x_grid)
    valleys, _ = find_peaks(-density)
    peaks, _ = find_peaks(density)
    return x_grid, density, peaks, valleys


def plot_analysis(x_grid, dists, density, peaks, valleys, divider) -> figure.Figure:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.hist(dists, density=True, alpha=0.8, label="Distances", color="#000")
    ax.plot(x_grid, density, lw=3, color="#e2001a", label="Kernel density est.")
    props = {"color": "#e2001a", "markersize": 8}
    ax.plot(x_grid[peaks], density[peaks], "D", label="Peaks (loc. max)", **props)
    ax.plot(x_grid[valleys], density[valleys], "s", label="Valleys (loc. min)", **props)
    ax.axvline(
        divider,
        color="black",
        linestyle="--",
        lw=1,
        label=f"Divider = {divider:.4f}",
    )
    ax.axvspan(x_grid.min(), divider, color="#e2001a", alpha=0.3)
    ax.set_xlabel("Cosine Distance")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)
    return fig


def show_subgroup(_, parent, namespace, query_fn):
    # st.subheader(f"{i}. {namespace.upper().replace('_', ' ')}", anchor=False)
    if namespace not in st.session_state:
        ids, metas, dists = query_fn(st.session_state["query_embed"], parent)
        x_grid, density, peaks, valleys = kde_analysis(dists)
        threshold = x_grid[valleys[0]] if len(valleys) else x_grid.max() + 1e-5
        select = [d < threshold for d in dists]
        select_ids = [id for (id, s) in zip(ids, select) if s]
        fig = plot_analysis(x_grid, dists, density, peaks, valleys, threshold)
        df = pd.DataFrame(
            {
                "Distance": dists,
                "Product": [
                    f"[{m['name']}]({m['url']})" if "url" in m else f"{m['name']}"
                    for m in metas
                ],
                "Select": ["__YES__" if s else "--" for s in select],
            }
        )
        st.session_state[namespace] = {
            "df": df,
            "ids": ids,
            "metas": metas,
            "dists": dists,
            "figure": fig,
            "select_ids": select_ids,
        }
    left_side, right_side = st.columns([1.4, 1])
    right_side.pyplot(st.session_state[namespace]["figure"], transparent=True)
    left_side.table(st.session_state[namespace]["df"])


def main():
    with st.form("query_form"):
        query = query_component()
        if query is None:
            return
    (
        st.session_state["query_text"],
        st.session_state["query_embed"],
        to_verify_results,
    ) = query
    tabs = st.tabs(
        [
            "__Category 1__",
            "__Category 2__",
            "__Category 3__",
            "__Product__",
            "__Matches__",
        ]
    )
    with tabs[0]:
        show_subgroup(1, None, "category", catalog.query_category)
    with tabs[1]:
        categories = st.session_state["category"]["select_ids"]
        show_subgroup(2, categories, "prod_family", catalog.query_prod_family)
    with tabs[2]:
        prod_families = st.session_state["prod_family"]["select_ids"]
        show_subgroup(3, prod_families, "prod_group", catalog.query_prod_group)
    with tabs[3]:
        prod_groups = st.session_state["prod_group"]["select_ids"]
        show_subgroup(4, prod_groups, "product", catalog.query_product)
    with tabs[4]:
        df = st.session_state["product"]["df"]
        df = df[df["Select"] == "__YES__"]
        if to_verify_results:
            with st.spinner("Verifying results...", show_time=True):
                df["Match"] = df["Product"].apply(
                    lambda x: "__YES__"
                    if is_match(st.session_state["query_text"], x)
                    else "NO"
                )
            st.table(df[["Distance", "Product", "Match"]])
        else:
            st.table(df[["Distance", "Product"]])


if __name__ == "__main__":
    main()
