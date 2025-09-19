import os

import streamlit as st

from catalog import Catalog

st.set_page_config(
    page_title="BEST AI Search Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .st-key-first_category { background-color: #d8d8d8; }
    .st-key-first_prod_family { background-color: #d8d8d8; }
    .st-key-first_prod_group { background-color: #d8d8d8; }
    .st-key-first_product { background-color: #d8d8d8; }
    </style>
    """,
    unsafe_allow_html=True,
)


def clear_state():
    keys_to_clear = [
        None,
        "category",
        "prod_family",
        "prod_group",
        "product",
        "query",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


page = "hierarchical"
if "page" in st.session_state and st.session_state["page"] != page:
    clear_state()
st.session_state["page"] = page


def select_category(cat, ns):
    st.session_state[ns]["top"] = cat
    if ns == "category":
        del st.session_state["prod_family"]
        del st.session_state["prod_group"]
        del st.session_state["product"]
    if ns == "prod_family":
        del st.session_state["prod_group"]
        del st.session_state["product"]
    if ns == "prod_group":
        del st.session_state["product"]


def new_query():
    for ns in ["category", "prod_family", "prod_group", "product"]:
        if ns in st.session_state:
            del st.session_state[ns]


def main():
    path = "catalog_db"
    catalog = Catalog(path)

    with st.form("query_form"):
        query_text = st.text_area(
            label="query",
            label_visibility="collapsed",
            placeholder="Enter query here...",
        )
        if st.form_submit_button("Submit", type="primary"):
            st.session_state["query"] = catalog.embed_document(query_text)
            new_query()
    if "query" not in st.session_state:
        return

    cols = st.columns([1, 1, 1, 1], border=True)
    for i, col, namespace, parent, qf in [
        (1, cols[0], "category", None, catalog.query_category),
        (2, cols[1], "prod_family", "category", catalog.query_prod_family),
        (3, cols[2], "prod_group", "prod_family", catalog.query_prod_group),
        (4, cols[3], "product", "prod_group", catalog.query_product),
    ]:
        with col:
            st.subheader(f"{i}. {namespace.upper().replace('_', ' ')}", anchor=False)
            st.markdown(" ")

            if namespace not in st.session_state:
                if namespace == "category":
                    ids, metas, dists = qf(st.session_state["query"])
                else:
                    ids, metas, dists = qf(
                        st.session_state["query"],
                        st.session_state[parent]["top"],
                    )
                st.session_state[namespace] = {
                    "top": ids[0],
                    "ids": ids,
                    "metadatas": metas,
                    "distances": dists,
                }

            for id, meta, dist in zip(
                st.session_state[namespace]["ids"],
                st.session_state[namespace]["metadatas"],
                st.session_state[namespace]["distances"],
            ):
                if id == st.session_state[namespace]["top"]:
                    key = f"first_{namespace}"
                else:
                    key = f"other_{namespace}_{id}"

                if namespace == "product":
                    name = meta["name"]
                    url = meta["url"] if "url" in meta else "https://www.best.cz/404"
                    c1, c2 = st.columns([1, 3])
                    c1.link_button(f"{dist:.4f}", url=url)
                    c2.markdown(f"[{name}]({url})")
                    continue

                with st.container(key=key):
                    c1, c2 = st.columns([1, 3])
                    c1.button(
                        f"{dist:.4f}",
                        on_click=select_category,
                        args=(id, namespace),
                        disabled=id == st.session_state[namespace]["top"],
                        key=f"button_{key}",
                    )
                    # c2.markdown(f"**[{id}]({meta['url']})**")
                    c2.markdown(f"**{id}**")


if __name__ == "__main__":
    main()
