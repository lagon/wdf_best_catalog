import os

import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
import streamlit as st

from catalog import Catalog

client = OpenAI()
model = "gpt-4.1"

st.set_page_config(
    page_title="BEST AI Search Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .st-key-first { background-color: #d8d8d8; }
    .st-key-dist-slider { padding: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def clear_state():
    keys_to_clear = [
        "query",
        "ids",
        "metadatas",
        "distances",
        "range",
        "distances",
        "query_text",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


page = "complete"
if "page" in st.session_state and st.session_state["page"] != page:
    clear_state()
st.session_state["page"] = page


def is_match(requested_desc: str, product_desc: str) -> bool:
    prompt = f"""
    Task: Compare two product descriptions for structural engineering items.
    If they somewhat match (correct type) return YES. Otherwise return NO.

    Requested product: {requested_desc}
    Catalog item: {product_desc}
    """
    resp = client.responses.create(
        model=model, input=prompt, temperature=0, top_p=1e-20
    )
    return "YES" in resp.output_text.upper()


def preprocess_query(query: str):
    prompt = f"""
    Process this technical construction item description into a natural language format.
    Keep essential product info for catalog matching and extend abbreviations.
    The output should be concise and ready-to-go for text embedding.

    Item: {query}
    """
    if not st.session_state["query_preprocess"]:
        return query
    return client.responses.create(
        model=model, input=prompt, temperature=0, top_p=1e-20
    ).output_text


def get_divider(data):
    data = np.array(data)
    x_grid = np.linspace(data.min(), data.max(), 1000)
    density = gaussian_kde(data)(x_grid)
    valleys, _ = find_peaks(-density)
    peaks, _ = find_peaks(density)
    return x_grid, density, peaks, valleys


def plot_analysis(x_grid, density, peaks, valleys):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.hist(
        st.session_state["distances"],
        bins=100,
        density=True,
        alpha=0.8,
        label="Distances",
    )
    ax.plot(x_grid, density, lw=3, color="#000", label="Kernel density est.")
    ax.plot(
        x_grid[peaks],
        density[peaks],
        "*",
        color="red",
        markersize=12,
        label="Peaks (loc. max)",
    )
    ax.plot(
        x_grid[valleys],
        density[valleys],
        "o",
        color="lime",
        markersize=10,
        label="Valleys (loc. min)",
    )
    if len(valleys):
        divider = x_grid[valleys[0]]
        ax.axvline(
            divider,
            color="black",
            linestyle="--",
            lw=1,
            label=f"Divider = {divider:.4f}",
        )
        ax.axvspan(ax.get_xlim()[0], divider, color="lightgreen", alpha=0.3)
    ax.set_xlabel("Cosine Distance")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig, transparent=True)


def plot_range_plot(x_grid, valleys):
    mn, mx = (
        min(st.session_state["distances"]) - 0.03,
        max(st.session_state["distances"]) + 0.03,
    )
    st.session_state["range"] = st.slider(
        "Distance",
        min_value=mn,
        max_value=mx,
        step=0.001,
        value=(
            (min(st.session_state["distances"]) + mn) / 2,
            float(x_grid[valleys[0]]) if len(valleys) else mx,
        ),
        label_visibility="hidden",
        key="dist-slider",
    )
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    ax.hist(st.session_state["distances"], bins=100, color="#e2001a")
    ax.set_xlabel("Distance")
    ax.set_xlim(mn, mx)
    ax.get_yaxis().set_visible(False)
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(False)
    x0, x1 = st.session_state["range"]
    d = {
        "alpha": 0.5,
        "color": "black",
        "edgecolor": None,
    }
    ax.axvspan(ax.get_xlim()[0], x0, **d)
    ax.axvspan(x1, ax.get_xlim()[1], **d)
    ax.axvline(x=x0, color="black", linewidth=2)
    ax.axvline(x=x1, color="black", linewidth=2)
    st.pyplot(fig, transparent=True)


def filter_dupl(dists, metas):
    dists2, metas2, memory = [], [], set()
    for d, m in zip(dists, metas):
        pair = (m["name"], m["url"])
        if pair not in memory:
            memory.add(pair)
            dists2.append(d)
            metas2.append(m)
    return dists2, metas2


def main():
    path = "catalog_db"
    catalog = Catalog(path)

    with st.form("query_form"):
        c1, c2 = st.columns([1, 1])
        query_text = c1.text_area(
            label="query",
            label_visibility="collapsed",
            placeholder="Enter query here...",
        )
        st.session_state["query_preprocess"] = st.checkbox(
            "LLM query preprocessing (experimental)", disabled=False
        )
        st.session_state["verify_results"] = st.checkbox(
            "LLM result verification (experimental)", disabled=False
        )
        if st.form_submit_button("Submit", type="primary") and query_text != "":
            clear_state()
            with st.spinner("Processing query...", show_time=True):
                st.session_state["query_text"] = preprocess_query(query_text)
                st.session_state["query"] = catalog.embed_document(
                    st.session_state["query_text"]
                )
        c2.text_area(
            label="processed_query",
            label_visibility="collapsed",
            disabled=True,
            value=st.session_state["query_text"]
            if "query_text" in st.session_state
            else "",
        )
    if "query" not in st.session_state:
        return
    if "ids" not in st.session_state:
        (
            st.session_state["ids"],
            st.session_state["metadatas"],
            st.session_state["distances"],
        ) = catalog.query_product(st.session_state["query"])

    x_grid, density, peaks, valleys = get_divider(st.session_state["distances"])

    col1, col2 = st.columns([1.5, 1], gap="medium", border=False)
    with col2:
        tab1, tab2 = st.tabs(["Filtering", "Analysis"])
        with tab1:
            plot_range_plot(x_grid, valleys)
        with tab2:
            plot_analysis(x_grid, density, peaks, valleys)
    with col1:
        dists = np.array(st.session_state["distances"])
        metas = st.session_state["metadatas"]
        for m in metas:
            if "url" not in m:
                m["url"] = None

        x0, x1 = st.session_state["range"]
        lt, gte = dists < x0, dists <= x1
        pre, pre_inc = sum(lt), sum(gte)
        dists = dists[pre:pre_inc]
        metas = metas[pre:pre_inc]
        dists, metas = filter_dupl(dists, metas)

        if st.session_state["verify_results"]:
            df = pd.DataFrame({"Distance": [], "Name": []})
            df.index += 1
            tab = st.table(df)

            with st.spinner("Verifying results...", show_time=True):
                for i, (d, m) in enumerate(zip(dists, metas)):
                    if i < 10:
                        if is_match(st.session_state["query_text"], m["text"]):
                            match = "__YES__"
                        else:
                            match = "NO"
                    else:
                        match = "_~_"

                    tab.add_rows(
                        {
                            "Distance": [f"{d:.4f}"],
                            "Name": [f"[{m['name']}]({m['url']})"],
                            "Match": [match],
                        }
                    )
        else:
            df = pd.DataFrame(
                {
                    "Distance": [f"{d:.4f}" for d in dists],
                    "Name": [f"[{m['name']}]({m['url']})" for m in metas],
                }
            )
            df.index += 1
            st.table(df)


if __name__ == "__main__":
    main()
