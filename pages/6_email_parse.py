import streamlit as st
import extract_msg
import re
import json
import dataclasses

st.set_page_config(
    page_title="BEST AI Search Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)


def clear_state():
    keys_to_clear = []
    for key in keys_to_clear:
        st.session_state.pop(key, None)


page = "email"
if "page" in st.session_state and st.session_state["page"] != page:
    clear_state()
st.session_state["page"] = page


@dataclasses.dataclass
class Person:
    name: str
    email: str


email_pattern = r"[a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z_-]{2,4}"


def extract_name(text, email):
    if text is None:
        return None
    pattern = rf"\s*(\w+\s\w+)\s*[<([](?:mailto:)?{email}"
    names = re.findall(pattern, text)
    names = list(set([n.strip() for n in names]))
    if len(names) == 1:
        return names[0]
    else:
        return None


def extract_emails(text):
    if text is None:
        return []
    email_list = re.findall(email_pattern, text)
    return [Person(extract_name(text, e), e) for e in list(set(email_list))]


# def sort_mail(pp):
#     order = [p for p in pp if p.name is not None and "best.cz" in p.name]
#     order += [p for p in pp if p.name is not None and "best.cz" not in p.name]
#     order += [p for p in pp if p.name is None]
#     return order


def sort_mail(pp):
    return sorted(pp, key=lambda p: (p.name is None, "best.cz" not in (p.email or "")))


def process_emails(files):
    email_dicts = []
    for file in files:
        msg = extract_msg.openMsg(file.read())
        data = json.loads(msg.getJson())  # ty: ignore
        email_dicts.append(
            {
                "name": file.name,
                "json": data,
            }
        )

    for d in email_dicts:
        d["from"] = extract_emails(d["json"]["from"])
        d["to"] = [
            *extract_emails(d["json"]["to"]),
            *extract_emails(d["json"]["cc"]),
            *extract_emails(d["json"]["bcc"]),
        ]
        d["all"] = d["to"] + [
            p
            for p in extract_emails(d["json"]["body"])
            if p.email not in [p.email for p in d["to"]]
        ]
        d["all"] = sort_mail(d["all"])
    return email_dicts


def main():
    with st.form("query_form"):
        # t1, t2 = st.tabs(["Upload file", "Input raw email"])
        files = st.file_uploader(
            "Nahrajte .msg soubor(y)",
            type="msg",
            accept_multiple_files=True,
            label_visibility="visible",
        )
        # t2.write("")
        # query_text = t2.text_area(
        #     label="query",
        #     placeholder="Zadejte obsah emailu...",
        # )
        if st.form_submit_button("Zpracovat", type="primary"):
            clear_state()
            with st.spinner("Analyzuji...", show_time=True):
                email_dicts = process_emails(files)
        else:
            return

    for d in email_dicts:
        st.markdown(f"### {d['name']}")
        c = st.container(border=True)
        c1, c2 = c.columns((2, 3))
        with c1:
            c1.markdown("**Zmíněné osoby:**")
            for p in d["all"]:
                if p.name:
                    st.markdown(
                        f"- {p.name} `{p.email}{' 💼' if 'best.cz' in p.email else ''}`"
                    )
                else:
                    st.markdown(f"- `{p.email}`")
        with c2.container(border=True):
            st.markdown(f"**Od:** {d['from'][0].name}")
            st.markdown(f"**Komu:** {', '.join([p.name for p in d['to']])}")
            st.markdown(f"**Předmět:** {d['json']['subject']}")
            st.markdown("---")
            c = st.container(border=False, height=300)
            c.caption(d["json"]["body"])
        # st.write(d["json"]["body"])


if __name__ == "__main__":
    main()
