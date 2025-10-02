import os
import json
import re
import typing as t

import config.configuration as cfg

COLOUR_SET = ["COLORMIX ARABICA", "COLORMIX BRILANT", "COLORMIX ETNA", "COLORMIX MOKA", "COLORMIX PODZIM",
              "COLORMIX SAHARA", "COLORMIX SAND", "ANTRACITOVÁ", "PŘÍRODNÍ", "KARAMELOVÁ", "ČERVENÁ",
              "PÍSKOVCOVÁ", "ČERVENOČERNÁ", "CIHLOVÁ", "CIHLOVOČERNÁ", "BÍLÁ", "TABARO", "TABELO", "TAGOLO",
              "TAMORO", "TOKANTO", "TOKARO", "VANILKOVÁ", "KRÉMOVÁ", "ŽLUTÁ", "VANTO", "VEGARO", "VELINO",
              "VERDO", "VERETO"]

def remove_colours_from_prod(product_name: str) -> str:
    colours = COLOUR_SET
    for c in colours:
        if c in product_name:
            product_name = product_name.replace(c, "")
            break
    return product_name.strip()

def remove_dimensions_from_prod(product_name: str) -> str:
    # regex_expr = re.compile("[0-9]+ cm", flags=re.IGNORECASE)
    regex_expr = re.compile("[0-9]{1,2}0{2} ?[X/] ?[0-9]{1,2}0{2}", flags=re.IGNORECASE)
    product_name = re.sub(regex_expr, "", product_name)
    regex_expr = re.compile(" [0-9]{1,4} ?CM", flags=re.IGNORECASE)
    product_name = re.sub(regex_expr, "", product_name)
    regex_expr = re.compile(" [0-9]+/[0-9]{1,4} ?CM", flags=re.IGNORECASE)
    product_name = re.sub(regex_expr, "", product_name)
    regex_expr = re.compile(" [S]?[0-9]{1,4}$", flags=re.IGNORECASE)
    product_name = re.sub(regex_expr, "", product_name)

    return product_name.strip()

def remove_roman_numerals_from_prod(product_name: str) -> str:
    roman_numerals = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII"]
    for rn in roman_numerals:
        regex_expr = re.compile(f" {rn}$", flags=re.IGNORECASE)
        product_name = re.sub(regex_expr, "", product_name)
    return product_name.strip()

def remove_keywords_from_prod(product_name: str) -> str:
    if product_name.startswith("BEST - MONO "):
        return "BEST - MONO"
    if product_name.startswith("BEST - RONDA "):
        return "BEST - RONDA"
    if product_name.startswith("BEST - ZASTÁVKOVÝ OBRUBNÍK "):
        return "BEST - ZASTÁVKOVÝ OBRUBNÍK"
    if product_name.startswith("BEST- ZÁSLEPKA UNIVERZÁLNÍ STROPNÍ VLOŽKY "):
        return "BEST - ZÁSLEPKA UNIVERZÁLNÍ STROPNÍ VLOŽKY"
    if product_name.startswith("BEST-UNIVERZÁLNÍ STROPNÍ VLOŽKA"):
        return "BEST-UNIVERZÁLNÍ STROPNÍ VLOŽKA"
    if product_name.startswith("LAPÁK TUKŮ"):
        return "LAPÁK TUKŮ"
    if product_name.startswith("TERČ K POKLÁDCE"):
        return "TERČ K POKLÁDCE"
    if product_name.startswith("BEST - KERBO"):
        return "BEST - KERBO"
    if product_name.startswith("BEST - LINEA "):
        return "BEST - LINEA"
    if product_name.startswith("BEST - PŘECHODOVÁ DESKA"):
        return "BEST - PŘECHODOVÁ DESKA"
    if product_name.startswith("BEST - SVODIDLO"):
        return "BEST - SVODIDLO"

    if product_name == "GIGANTICKÁ":
        return "BEST - GIGANTICKÁ"
    if product_name == "LINEA":
        return "BEST - LINEA"
    if product_name == "BEST - ŽELEZOB.TROUBA/TZP-Q 1250/1000":
        return "BEST - ŽELEZOB.TROUBA/TZP-Q"

    return product_name

def remove_colours_from_all_products(all_product_titles: t.List[str]) -> t.Dict[str, t.List[str]]:
    updated_prod_names: t.Dict[str, t.List[str]] = {}

    for title in all_product_titles:
        base_name = remove_colours_from_prod(title)
        base_name = remove_dimensions_from_prod(base_name)
        base_name = remove_roman_numerals_from_prod(base_name)
        base_name = remove_keywords_from_prod(base_name)
        if base_name not in updated_prod_names:
            updated_prod_names[base_name] = []
        updated_prod_names[base_name].append(title)

    return updated_prod_names

if __name__ == '__main__':
    with open(os.path.join(cfg.get_settings().main_results_path_dir, "all_products.json"), "rt", encoding="utf-8") as f:
        all_products = json.load(f)

    all_product_titles: t.List[str] = []
    for prod in all_products:
        all_product_titles.append(prod["product_title"])

    base_product_names: t.Dict[str, t.List[str]] = remove_colours_from_all_products(all_product_titles)

    with open(os.path.join(cfg.get_settings().main_results_path_dir, "base_product_titles.txt"), "wt", encoding="utf-8") as f:
        sorted_keys = sorted(base_product_names.keys())
        for key in sorted_keys:
            f.write(f"{key}\n")
            for v in base_product_names[key]:
                f.write(f"\t{v}\n")

    with open(os.path.join(cfg.get_settings().main_results_path_dir, "base_product_titles.json"), "wt", encoding="utf-8") as f:
        json.dump(base_product_names, f, ensure_ascii=False, indent=4)
