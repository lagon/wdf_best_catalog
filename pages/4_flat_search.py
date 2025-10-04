import streamlit as st

from flat_catalog import Collection

st.set_page_config(
    page_title="BEST AI Search Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)


def clear_state():
    keys_to_clear = [
        "query",
        "choice",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


page = "flat"
if "page" in st.session_state and st.session_state["page"] != page:
    clear_state()
st.session_state["page"] = page


def main():
    groups = Collection("catalog_db2/groups.npz")
    products = Collection("catalog_db2/products.npz")
    colors = Collection("catalog_db2/colors.npz")

    with st.form("query_form"):
        query_text = st.text_area(
            label="query",
            label_visibility="collapsed",
            placeholder="Enter query here...",
        )
        if st.form_submit_button("Submit", type="primary") and query_text != "":
            clear_state()
            with st.spinner("Processing query...", show_time=True):
                st.session_state["query"] = groups.embedder(query_text)
    if "query" not in st.session_state:
        return
    metas, dists = groups._search(st.session_state["query"])
    colors, col_dists = colors._search(st.session_state["query"])
    products, prod_dists = products._search(st.session_state["query"])

    c1, c2 = st.columns([1, 3], gap="small")
    with c2.container(border=True):
        if "choice" not in st.session_state:
            st.session_state["choice"] = 0
        m = metas[st.session_state["choice"]]
        st.markdown(f"### {m['name']}")

        # sort products and colors by col_dists and prod_dists
        ps, cs = [], []
        for p, d in zip(products, prod_dists):
            if p["name"] in m["products"]:
                ps.append((d, f"[{p['name']}]({p['url']})"))
        for c, d in zip(colors, col_dists):
            if c in m["colors"]:
                cs.append((d, c))

        # col1, col2 = st.columns([1, 1], gap="medium")
        # col1.markdown("#### Products")
        # col1.table(ps)
        # col2.markdown("#### Colors")
        # col2.table(cs)
        st.markdown("#### Products")
        st.table(ps)

        st.markdown("#### Description")
        st.markdown(m["description"])

    for i, m in enumerate(metas):
        c1.button(
            f"__{m['name']}__",
            # f"{dists[i]:.4f} \n\n __{m['name']}__",
            help=f"Similarity: {dists[i]:.6f}",
            width="stretch",
            type="primary" if i == st.session_state["choice"] else "secondary",
            on_click=lambda idx=i: st.session_state.update({"choice": idx}),
        )


if __name__ == "__main__":
    main()
