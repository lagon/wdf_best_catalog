import json
import os
import typing as t
import sys

import streamlit as st
import openai as oai

import oai_batch

import alternate_hierarchy.open_llm_resolver_v6 as v6

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

def chatbot_resolver(query_text: str, alt_hierarchy_db: t.Dict, prog_bar):
    assert os.environ.get("OPENAI_API_KEY") is not None
    client: oai.OpenAI = oai.OpenAI()

    request_offer_list = [{
        "CUSTOMER DESCRIPTION": query_text,
        "OFFERED PRODUCT NAME": "NA"
    }]

    prog_bar.progress(20, text="Generuji přepis produktu")
    batch: oai_batch.OAI_Worker = oai_batch.OAI_Direct(client=client, number_active_batches=100, max_work_items_to_add=100, tag="extend_desc", working_dir="tmp")
    desc_expansion: t.List[t.Dict] = v6.expand_customer_requested_product(request_offer_list=request_offer_list, batch=batch, desc_expansion_model_name="gpt-4.1-mini", work_dir="tmp")
    # desc_expansion: t.List[t.Dict] = v6.expand_customer_requested_product(request_offer_list=request_offer_list, batch=batch, desc_expansion_model_name="gpt-5", work_dir="tmp")

    prog_bar.progress(40, text="Hledám typ produktu")
    batch: oai_batch.OAI_Worker = oai_batch.OAI_Direct(client=client, number_active_batches=100, max_work_items_to_add=100, tag="find_prod_type", working_dir="tmp")
    output =v6.find_product_type(desc_expansion=desc_expansion, alt_hierarchy_db=alt_hierarchy_db, batch=batch, desc_expansion_model_name="gpt-4.1-mini", work_dir="tmp")
    selected_product_types = output["match"]

    prog_bar.progress(60, text="Hledám skupinu produktů")
    batch: oai_batch.OAI_Worker = oai_batch.OAI_Direct(client=client, number_active_batches=100, max_work_items_to_add=100, tag="find_prod_group", working_dir="tmp")
    output = v6.find_product_group(selected_product_type=selected_product_types, alt_hierarchy_db=alt_hierarchy_db, batch=batch, desc_expansion_model_name="gpt-4.1-mini", work_dir="tmp")
    selected_product_groups = output["match"]

    prog_bar.progress(80, text="Hledám produkt")
    batch: oai_batch.OAI_Worker = oai_batch.OAI_Direct(client=client, number_active_batches=100, max_work_items_to_add=100, tag="find_prod", working_dir="tmp")
    output = v6.find_products(selected_product_groups=selected_product_groups, alt_hierarchy_db=alt_hierarchy_db, batch=batch, desc_expansion_model_name="gpt-4.1-mini", work_dir="tmp")
    selected_products = output["match"]

    prog_bar.progress(100, text="Hotovo")
    selected_products.sort(key=lambda v: v["PRODUCT PCT"], reverse=True)
    return selected_products

def main():
    if "alt_hierarchy_db" not in st.session_state:
        with open("data/alternative_hierarchy_db.json", "rt", encoding="utf-8") as f:
            st.session_state["alt_hierarchy_db"] = json.load(f)

    with st.form("query_form"):
        query_text = st.text_area(
            label="query",
            label_visibility="collapsed",
            placeholder="Enter query here...",
        )

        if st.form_submit_button("Submit", type="primary") and query_text != "":
            clear_state()
            prog_bar = st.progress(value=0, text="Looking for products...")

            with st.spinner("Processing query...", show_time=True):
                st.session_state["response"] = chatbot_resolver(query_text, st.session_state["alt_hierarchy_db"], prog_bar)

    if "response" not in st.session_state:
        return

    if len(st.session_state["response"]) == 0:
        st.markdown(f"No results found for query '{query_text}'")
        return

    if "choice" not in st.session_state:
        st.session_state["choice"] = 0

    alt_hierarchy_db = st.session_state["alt_hierarchy_db"]
    selected = st.session_state["response"][st.session_state["choice"]]
    prod_info = alt_hierarchy_db["product_db"][selected['PRODUCT']]

    with st.popover(label="Přepis produktu"):
        st.markdown(f"#### Přepis produktu")
        st.markdown(f"{selected["EXTENDED DESCRIPTION"]}")

    c1, c2 = st.columns([1, 3], gap="small")
    with c2.container(border=True):
        st.markdown(f"### {selected['PRODUCT']} ({selected['PRODUCT PCT']})")
        st.markdown(f"__Odkaz__ na stránku [produktu]({prod_info['product_url']})")

        st.markdown(f"#### Generované shrnutí popisu produktu")
        st.markdown(f"{prod_info['product_summary']}")

        st.markdown(f"#### Popis produktu z webu")
        st.markdown(f"{prod_info['product_short_description']}")

        st.markdown(f"#### Rozšířený popis produktu z webu")
        st.markdown(f"{prod_info['product_details_description']}")

        st.markdown(f"#### Tabulka rozměrů")
        st.markdown(f"{prod_info['product_table_details']}")

        st.divider()

        st.markdown(f"#### Zařazení produtu s popisy")
        st.markdown(f"""
 * [__{prod_info['category']}__]({prod_info['category_url']}) - {prod_info['category_description']}
 * [__{prod_info['prod_family']}__]({prod_info['prod_family_url']}) - {prod_info['prod_family_description']}
 * [__{prod_info['product_group']}__]({prod_info['product_group_url']}) - {prod_info['product_group_description']}
 * [__{prod_info['product_title']}__]({prod_info['product_url']}) - {prod_info['product_summary']}
""")

    for i, m in enumerate(st.session_state["response"]):
        c1.button(
            label=f"__{m['PRODUCT']}__ ({m['PRODUCT PCT']})",
            help=f"Fit: {m['PRODUCT PCT']}",
            width="stretch",
            type="primary" if i == st.session_state["choice"] else "secondary",
            on_click=lambda idx=i: st.session_state.update({"choice": idx}),
        )

if __name__ == "__main__":
     main()
