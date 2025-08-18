import hashlib
import json
import os
import typing as t

import tqdm
from bs4 import BeautifulSoup
import requests
import urllib.parse as urlparse


def download_website(url: str) -> t.Optional[str]:
    cache_path = os.path.join("/home/cepekmir/Sources/wdf_best_catalog/tmp", hashlib.md5(url.encode()).hexdigest())
    if os.path.exists(cache_path):
        with open(cache_path, "rt") as f:
            return f.read()
    else:
        response = requests.get(url)
        if response.status_code == 200:
            with open(cache_path, "wt") as f:
                f.write(response.text)
            return response.text
        else:
            return None

def extract_links_from_category_page(webpage: str, web_url: str, base_url: str) -> t.Dict[str, t.Union[t.List[str], str]]:
    # Parse with BeautifulSoup
    soup = BeautifulSoup(webpage, "html.parser")

    # Extract all links from them
    links = []
    category_title = soup.find(name="div", class_="supercategory__left").find(name="h1").text
    category_description = soup.find(name="div", class_="supercategory__left--description").text.strip()

    # Find all divs with class 'card-product'
    card_divs = soup.find_all("div", class_="supercategory__whitebox--banners d-flex")
    for div in card_divs:
        for a in div.find_all("a", href=True):
            href = a["href"]
            href = urlparse.urljoin(base_url, href)
            links.append(href)

    # Remove duplicates if needed
    links = list(set(links))

    return {
        "category": category_title,
        "category_description": category_description,
        "category_url": web_url,
        "links": links
    }

def extract_links_from_prod_family_page(webpage: str, web_url: str, base_url: str) -> t.Dict[str, t.Union[t.List[str], str]]:
    soup = BeautifulSoup(webpage, "html.parser")
    # Extract all links from them
    links = set()

    prod_fam_div = soup.find(name="div", class_="category__left")
    if prod_fam_div is None:
        return {}

    prod_family_title = prod_fam_div.find(name="h1").text
    prod_family_description = soup.find(name="div", class_="category__left--description").text.strip()

    prod_family_list = soup.find(name="div", id="category-list__content")
    for link in prod_family_list.find_all("a", href=True):
        links.add(link["href"])

    return {
        "prod_family": prod_family_title,
        "prod_family_description": prod_family_description,
        "prod_family_url": web_url,
        "prod_family_links": list(links)
    }

def is_product_page(webpage: str) -> bool:
    soup = BeautifulSoup(webpage, "html.parser")
    category_list_grid = soup.find(name="div", id="category-list__content")
    return category_list_grid is None

def extract_products_from_product_groups(webpage: str, web_url, base_url: str) -> t.List[t.Dict[str, t.Any]]:
    soup = BeautifulSoup(webpage, "html.parser")
    category_list_content_list = soup.find(name="div", id="category-list__content", class_="product-grid")
    product_cards = category_list_content_list.find_all(name="div", class_="card-product")
    prod_subgroup_title = soup.find(name="div", class_="category__left").find(name="h1").text
    prod_subgroup_description = soup.find(name="div", class_="category__left--description").text.strip()

    products = []
    for card in product_cards:
        product_link = card.find(name="a", href=True)["href"]
        if product_link == web_url:
            print("Self link!")
        else:
            products.append({
                "product_group": prod_subgroup_title,
                "product_group_description": prod_subgroup_description,
                "product_group_url": web_url,
                "product_url": product_link,
                "is_product": False
            })
    return products

def extract_details_from_prod_page(webpage: str, web_url: str) -> t.Dict[str, t.Union[t.List[str], str]]:
    soup = BeautifulSoup(webpage, "html.parser")
    # Extract all links from them
    # print(web_url)

    prod_detail_div = soup.find(name="div", class_="product-detail__header")
    if prod_detail_div is None:
        if web_url == "https://www.best.cz/ulicni-destove-vpusti-dn450":
            return {
                "product_title": "Uliční vpusti DN 450",
                "product_short_description": "Uliční vpusti: Nezbytný prvek pro odvod dešťových vod",
                "product_details_description": """Uliční vpusti: Nezbytný prvek pro odvod dešťových vod
Uliční vpusti jsou klíčovým prvkem městské infrastruktury, zajišťujícím efektivní zachycování a odvádění dešťových vod z pozemních komunikací a dalších veřejných prostranství. Tyto konstrukce jsou navrženy tak, aby minimalizovaly riziko zaplavení a zajistily plynulý odtok vody do stokové sítě, čímž přispívají k ochraně životního prostředí a zlepšení kvality života ve městech.

Konstrukce a funkce uličních vpustí

Hlavním účelem uličních vpustí je shromažďování dešťových vod a jejich následný odvod do kanalizačního systému. Aby byla zajištěna maximální účinnost a spolehlivost, jsou uliční vpusti často vybaveny lapačem nečistot, známým také jako kalový koš. Tento prvek slouží k zachycování hrubých nečistot, jako je štěrk, listí nebo jiné organické materiály, které by mohly způsobit ucpání odtokového systému. Kalové koše mohou být navrženy buď s kalovou prohlubní, která umožňuje usazování nečistot na dně vpusti, nebo s odtokem ve spodní části, což zajišťuje plynulý průtok vody a minimalizuje riziko ucpání. Tato kombinace zaručuje, že i při intenzivních srážkách je voda účinně odváděna z povrchu komunikací a veřejných prostranství.

Materiály a normy

Pro výrobu uličních vpustí se používá beton třídy C35/45, který splňuje vysoké požadavky na odolnost a pevnost. Tento beton je navržen dle normy ČSN EN 1917:2004, která specifikuje složení betonu pro stupeň vlivu prostředí XF4. To znamená, že beton je odolný vůči mrazu a rozmrazování, což je klíčové pro zajištění dlouhé životnosti a funkčnosti uličních vpustí v náročných klimatických podmínkách.

Význam uličních vpustí v městské infrastruktuře

Uliční vpusti hrají zásadní roli v prevenci povodní a ochraně městské infrastruktury. Efektivní odvádění dešťových vod pomáhá předcházet erozi, poškození silnic a chodníků, a také minimalizuje riziko vzniku stagnačních vod, které mohou být zdrojem nepříjemných zápachů a líhní komárů. Díky pečlivému návrhu a použití kvalitních materiálů představují uliční vpusti spolehlivé řešení pro moderní města, které čelí výzvám spojeným s klimatickými změnami a rostoucím počtem srážek. Zajišťují tak nejen ochranu infrastruktury, ale také přispívají k udržitelnosti a komfortu městského prostředí.""",
                "product_table_details": """
"dna skruží DN 450"
[{"název": "TBV-Q 1a/450/330 dno s výt. DN 150 PVC",	"D (mm)" :450,	"H (mm)" :330,	"t (mm)" :"-",	"hmotnost ks (kg)" :83,	"počet ks na paletě" :16},
{"název": "TBV-Q 1d/450/380 dno s výt. DN 200 PVC",	"D (mm)" :450,	"H (mm)" :380,	"t (mm)" :"-",	"hmotnost ks (kg)" :87,	"počet ks na paletě" :12},
{"název": "TBV-Q 1d/450/330 dno s výt. DN 200 bez vložky",	"D (mm)" :450,	"H (mm)" :380,	"t (mm)" :"-",	"hmotnost ks (kg)" :84,	"počet ks na paletě" :12},
{"název": "TBV-Q 2a/450/300 dno s kalovou prohlubní",	"D (mm)" :450,	"H (mm)" :300,	"t (mm)" :"-",	"hmotnost ks (kg)" :71,	"počet ks na paletě" :16}]
 

"skruže DN 450"
[{"název": "TBV-Q 5c/450/195 skruž horní",	"D (mm)" :450,	"H (mm)" :195,	"t (mm)" :50,	"hmotnost ks (kg)" :38,	"počet ks na paletě" :20},
{"název": "TBV-Q 5b/450/295 skruž horní",	"D (mm)" :450,	"H (mm)" :295,	"t (mm)" :50,	"hmotnost ks (kg)" :57,	"počet ks na paletě" :16},
{"název": "TBV-Q 5d/450/570 skruž horní",	"D (mm)" :450,	"H (mm)" :570,	"t (mm)" :50,	"hmotnost ks (kg)" :105,	"počet ks na paletě" :8},
{"název": "TBV-Q 6b/450/195 skruž středová",	"D (mm)" :450,	"H (mm)" :195,	"t (mm)" :50,	"hmotnost ks (kg)" :38,	"počet ks na paletě" :24},
{"název": "TBV-Q 6a/450/295 skruž středová",	"D (mm)" :450,	"H (mm)" :295,	"t (mm)" :50,	"hmotnost ks (kg)" :56,	"počet ks na paletě" :16},
{"název": "TBV-Q 6d/450/570 skruž středová",	"D (mm)" :450,	"H (mm)" :570,	"t (mm)" :50,	"hmotnost ks (kg)" :105,	"počet ks na paletě" :8},
{"název": "TBV-Q 3a/450/350 skruž s výtokem DN 150 PVC",	"D (mm)" :450,	"H (mm)" :350,	"t (mm)" :50,	"hmotnost ks (kg)" :75,	"počet ks na paletě" :16},
{"název": "TBV-Q 3a/450/350 skruž s výtokem DN 200 PVC",	"D (mm)" :450,	"H (mm)" :350,	"t (mm)" :50,	"hmotnost ks (kg)" :70,	"počet ks na paletě" :16},
{"název": "TBV-Q 3d/450/450 skruž s výtokem DN 200 bez vložky",	"D (mm)" :450,	"H (mm)" :450,	"t (mm)" :50,	"hmotnost ks (kg)" :90,	"počet ks na paletě" :8},
{"název": "TBV-Q S/450/550 skruž se zápachovou uzávěrkou DN 150 PVC",	"D (mm)" :450,	"H (mm)" :550,	"t (mm)" :50,	"hmotnost ks (kg)" :180,	"počet ks na paletě" :4},
{"název": "TBV-Q S/450/550 skruž se zápachovou uzávěrkou DN 200 PVC",	"D (mm)" :450,	"H (mm)" :550,	"t (mm)" :50,	"hmotnost ks (kg)" :190,	"počet ks na paletě" :4},
{"název": "TBV-Q 11/325 kónus",	"D (mm)" :450/270",	"H (mm)" :325,	"t (mm)" :50,	"hmotnost ks (kg)" :60,	"počet ks na paletě": 12}] 

"prstence DN 450"
[{"název": "TBV-Q 10a/627/390/60",	"D (mm)" :390,	"H (mm)" :60,	"t (mm)" :"-",	"hmotnost ks (kg)" :23,	"počet ks na paletě" :15},
{"název": "TBV-Q 10b/500x350/400x2720/60",	"D (mm)" :400/270,	"H (mm)" :60,	"t (mm)" :"-",	"hmotnost ks (kg)" :8,	"počet ks na paletě" :50}]
"""
            }
        if web_url == "https://www.best.cz/ulicni-destove-vpusti-dn500":
            return {
                "product_title": "Uliční vpusti DN 500",
                "product_short_description": "Uliční vpusti: Nezbytný prvek pro odvod dešťových vod",
                "product_details_description": """Uliční vpusti: Nezbytný prvek pro odvod dešťových vod
Uliční vpusti jsou klíčovým prvkem městské infrastruktury, zajišťujícím efektivní zachycování a odvádění dešťových vod z pozemních komunikací a dalších veřejných prostranství. Tyto konstrukce jsou navrženy tak, aby minimalizovaly riziko zaplavení a zajistily plynulý odtok vody do stokové sítě, čímž přispívají k ochraně životního prostředí a zlepšení kvality života ve městech.

Konstrukce a funkce uličních vpustí

Hlavním účelem uličních vpustí je shromažďování dešťových vod a jejich následný odvod do kanalizačního systému. Aby byla zajištěna maximální účinnost a spolehlivost, jsou uliční vpusti často vybaveny lapačem nečistot, známým také jako kalový koš. Tento prvek slouží k zachycování hrubých nečistot, jako je štěrk, listí nebo jiné organické materiály, které by mohly způsobit ucpání odtokového systému.

Kalové koše mohou být navrženy buď s kalovou prohlubní, která umožňuje usazování nečistot na dně vpusti, nebo s odtokem ve spodní části, což zajišťuje plynulý průtok vody a minimalizuje riziko ucpání. Tato kombinace zaručuje, že i při intenzivních srážkách je voda účinně odváděna z povrchu komunikací a veřejných prostranství.

Materiály a normy

Pro výrobu uličních vpustí se používá beton třídy C35/45, který splňuje vysoké požadavky na odolnost a pevnost. Tento beton je navržen dle normy ČSN EN 1917:2004, která specifikuje složení betonu pro stupeň vlivu prostředí XF4. To znamená, že beton je odolný vůči mrazu a rozmrazování, což je klíčové pro zajištění dlouhé životnosti a funkčnosti uličních vpustí v náročných klimatických podmínkách.

Význam uličních vpustí v městské infrastruktuře

Uliční vpusti hrají zásadní roli v prevenci povodní a ochraně městské infrastruktury. Efektivní odvádění dešťových vod pomáhá předcházet erozi, poškození silnic a chodníků, a také minimalizuje riziko vzniku stagnačních vod, které mohou být zdrojem nepříjemných zápachů a líhní komárů.

Díky pečlivému návrhu a použití kvalitních materiálů představují uliční vpusti spolehlivé řešení pro moderní města, které čelí výzvám spojeným s klimatickými změnami a rostoucím počtem srážek. Zajišťují tak nejen ochranu infrastruktury, ale také přispívají k udržitelnosti a komfortu městského prostředí.""",
                "product_table_details": """dna DN 500
"název": TBV-Q 500/190 D",	"D (mm)" :"500",	H (mm) :"190",	"t (mm)" :"50",	"hmotnost ks (kg)" :"78"
"název": TBV-Q 500/626 D",	"D (mm)" :"500",	"H (mm)" :"656",	"t (mm)" :"50",	"hmotnost ks (kg)" :"175"
"název": TBV-Q 500/626/200 VD",	"D (mm)" :"500",	"H (mm)" :"626",	"t (mm)" :"50",	"hmotnost ks (kg)" :"232"
"název": TBV-Q 500/626/150 VVD",	"D (mm)" :"500",	"H (mm)" :"626",	"t (mm)" :"50",	"hmotnost ks (kg)" :"232"
"název": TBV-Q 500/626/200 VVD",	"D (mm)" :"500",	"H (mm)" :"626",	"t (mm)" :"50",	"hmotnost ks (kg)" :"232"

skruže DN 500
"název": TBV-Q 500/590/200 V",	"D (mm)" :"500",	"H (mm)" :"590",	"t (mm)" :"50",	"hmotnost ks (kg)" :"170"
"název": TBV-Q 500/590/150 VV",	"D (mm)" :"500",	"H (mm)" :"590",	"t (mm)" :"50",	"hmotnost ks (kg)" :"170"
"název": TBV-Q 500/590/200 VV",	"D (mm)" :"500",	"H (mm)" :"590",	"t (mm)" :"50",	"hmotnost ks (kg)" :"170"
"název": TBV-Q 500/190",	"D (mm)" :"500",	"H (mm)" :"190",	"t (mm)" :"50",	"hmotnost ks (kg)" :"40"
"název": TBV-Q 500/290",	"D (mm)" :"500",	"H (mm)" :"290",	"t (mm)" :"50",	"hmotnost ks (kg)" :"60"
"název": TBV-Q 500/590",	"D (mm)" :"500",	"H (mm)" :"590",	"t (mm)" :"50",	"hmotnost ks (kg)" :"120"
"název": TBV-Q 500/290 K",	"D (mm)" :"500",	"H (mm)" :"290",	"t (mm)" :"50",	"hmotnost ks (kg)" :"87"

prstence DN 500
"název": TBV-Q 390/60",	"H (mm)" :"390",	"H (mm)" :"60",	"t (mm)" :"235/85",	"hmotnost ks (kg)" :"64"
"název": TBV-Q 660/180",	"H (mm)" :"660",	"H (mm)" :"180",	"t (mm)" :"100",	"hmotnost ks (kg)" :"103"
"název": TBV-Q 660/180/111 S",	"H (mm)" :"660",	"H (mm)" :"180/111",	"t (mm)" :"100",	"hmotnost ks (kg)" :"85"
"""
            }

        return {
            "product_title": "",
            "product_short_description": "",
            "product_details_description": "",
            "product_table_details": ""
        }

    product_title = prod_detail_div.find(name="h1").text
    product_short_description = soup.find(name="div", class_="product-detail-cart__text").text.strip()
    product_details_description = soup.find(name="div", class_="product-detail-description__content").text.strip()

    table_details = extract_table_details(webpage, product_title)

    return {
        "product_title": product_title,
        "product_short_description": product_short_description,
        "product_details_description": product_details_description,
        "product_table_details": table_details
    }

def _row_to_formated_cells(row, table_tag):
    headers = []
    for hdr in row.find_all(name=table_tag):
        colspan = 1
        rowspan = 1
        if "colspan" in hdr.attrs:
            colspan = int(hdr.attrs["colspan"])
        if "rowspan" in hdr.attrs:
            rowspan = int(hdr.attrs["rowspan"])

        for _ in range(colspan):
            h_info = {"name": hdr.text.strip(), "rowspan": rowspan}
            headers.append(h_info)
    return headers

def _regularise_table(content_rows):
    if len(content_rows) == 0:
        return  []
    for col_id in range(len(content_rows[0])):
        for row_id in range(len(content_rows)):
            if row_id+1 >= len(content_rows):
                continue
            if content_rows[row_id][col_id]["rowspan"] > 1:
                content_rows[row_id + 1].insert(col_id, {
                    "name": content_rows[row_id][col_id]["name"],
                    "rowspan": content_rows[row_id][col_id]["rowspan"] - 1,
                })
                content_rows[row_id][col_id]["rowspan"] = 1
    return content_rows

def extract_table_details(webpage: str, product_title: str) -> t.List[t.Dict[str, t.Any]]:
    try:
        soup = BeautifulSoup(webpage, "html.parser")
        details_table_div = soup.find(name="div", class_="product-detail-description__content")
        details_table = details_table_div.find(name="tbody")
        if details_table is None:
            return []
        rows = details_table.find_all(name="tr")

        last_row_cell = rows[-1].find("td")
        if ("colspan" in last_row_cell.attrs) and (last_row_cell.attrs["colspan"] == "14"):
            rows = rows[:-2]

        headers_all_rows = []
        content_all_rows = []

        for row in rows:
            headers = _row_to_formated_cells(row, table_tag="th")
            if len(headers) > 0:
                headers_all_rows.append(headers)

        for row in rows:
            content = _row_to_formated_cells(row, table_tag="td")
            if len(content) > 0:
                content_all_rows.append(content)

        # if len(headers_all_rows) == 0 or len(content_all_rows) == 0:
        #     return []

        headers_all_rows = _regularise_table(headers_all_rows)
        content_all_rows = _regularise_table(content_all_rows)

        if len(headers_all_rows) == 0:
            if len(content_all_rows) > 2:
                headers_all_rows = content_all_rows[:2]
                content_all_rows = content_all_rows[2:]
            else:
                return []

        header_labels = []
        for col_id in range(len(headers_all_rows[0])):
            col_txt = "+".join([headers_all_rows[i][col_id]["name"] for i in range(len(headers_all_rows))])
            if (len(header_labels) == 0) or (col_txt != header_labels[-1]):
                header_labels.append(col_txt)

        parsed_rows = []
        for row in content_all_rows:
            parsed_row = {}
            for lbl, cell in zip(header_labels, row):
                parsed_row[lbl] = cell["name"]
            parsed_rows.append(parsed_row)

        return parsed_rows
    except Exception as e:
        print(f"Failed to parse details table with error {e}")
        return []

def _join_labels(l1: str, l2: str) -> str:
    if l1 == l2:
        return l1
    if l1 == "":
        return l2
    if l2 == "":
        return l1
    return f"{l1} & {l2}"


if __name__ == '__main__':

    categories_urls = [
        "https://www.best.cz/dlazby",
        "https://www.best.cz/obrubniky",
        "https://www.best.cz/ploty-a-zdi",
        "https://www.best.cz/schodiste-a-palisady",
        "https://www.best.cz/ztracene-bedneni",
        "https://www.best.cz/studny-a-kanalizace",
        "https://www.best.cz/dopravni-infrastruktura",
        "https://www.best.cz/hruba-stavba"
    ]


    all_prod_families: t.List[t.Dict] = []
    for cat_url in tqdm.tqdm(categories_urls, desc="Getting product families in categories", ncols=100):
        webtxt = download_website(url=cat_url)
        if webtxt is None:
            print(f"   !!!! Failed to download {cat_url}. Skipping.")
            continue
        product_family = extract_links_from_category_page(webpage=webtxt, web_url=cat_url, base_url="https://www.best.cz/")
        all_prod_families.append(product_family)
    # for pf in all_prod_families:
    #     print(f" +++ {pf}")

    all_product_groups: t.List[t.Dict] = []
    for pf in tqdm.tqdm(all_prod_families, desc="Getting product groups in product families", ncols=100):
        for link in pf["links"]:
            webtxt = download_website(link)
            if webtxt is None:
                print(f"   !!!! Failed to download {pf['category_url']}. Skipping.")
                continue
            prod_group = extract_links_from_prod_family_page(webpage=webtxt, web_url=link, base_url="https://www.best.cz/")
            if len(prod_group) == 0:
                continue
            prod_group["category"] = pf["category"]
            prod_group["category_description"] = pf["category_description"]
            prod_group["category_url"] = pf["category_url"]

            all_product_groups.append(prod_group)

    all_product_groups.sort(key=lambda x: x["prod_family_url"], reverse=True)
    # for pg in all_product_groups:
    #     print(f" >>> {pg}")


    all_products: t.List[t.Dict] = []


    for pg in tqdm.tqdm(all_product_groups, desc="Getting product sub-groups in product groups", ncols=100):
        pg_links = pg["prod_family_links"]
        pg_links.sort()
        prods_in_subgroup = []
        for link in pg_links:
            webtxt = download_website(link)
            if webtxt is None:
                print(f"   !!!! Failed to download {pg['prod_family_url']}. Skipping.")
                continue
            if is_product_page(webpage=webtxt):
                product_dict = {
                    "product_url": link,
                    "product_group": pg["prod_family"],
                    "product_group_description": "",
                    "product_group_url": link,
                    "is_product": True
                }
                prods_in_subgroup.append(product_dict)
            else:
                prods_in_subgroup.extend(extract_products_from_product_groups(webpage=webtxt, web_url=link, base_url="https://www.best.cz/"))

        for p_sg in prods_in_subgroup:
            p_sg["category"] = pg["category"]
            p_sg["category_description"] = pg["category_description"]
            p_sg["category_url"] = pg["category_url"]
            p_sg["prod_family"] = pg["prod_family"]
            p_sg["prod_family_description"] = pg["prod_family_description"]
            p_sg["prod_family_url"] = pg["prod_family_url"]
        all_products.extend(prods_in_subgroup)


    # all_products = [{
    #     "product_url": "https://www.best.cz/inbelisima-dreno/antracitova/INBELISIMA8D05",
    #     "is_product": True
    # }]

    finished_products = []
    while len(all_products) > 0:
        rexamine_products = []
        products_with_info = []

        for prod in tqdm.tqdm(all_products, desc="Getting information about all products", ncols=100):
            link = prod["product_url"]
            # print(link)
            webtxt = download_website(link)
            if webtxt is None:
                print(f"   !!!! Failed to download {link}. Skipping.")
                continue

            if not prod["is_product"]:
                rex_prods = []
                if is_product_page(webpage=webtxt):
                    product_dict = {
                        "product_url": link,
                        "product_group": prod["product_group"] if "product_group" in prod else "NA",
                        "product_group_description": prod["product_group_description"] if "product_group_description" in prod else "",
                        "product_group_url": link,
                        "is_product": True
                    }
                    rex_prods.append(product_dict)
                else:
                    rex_prods.extend(extract_products_from_product_groups(webpage=webtxt, web_url=link, base_url="https://www.best.cz/"))
                for rp in rex_prods:
                    rp["category"] = prod["category"]
                    rp["category_description"] = prod["category_description"]
                    rp["category_url"] = prod["category_url"]
                    rp["prod_family"] = prod["prod_family"]
                    rp["prod_family_description"] = prod["prod_family_description"]
                    rp["prod_family_url"] = prod["prod_family_url"]
                    # rp["product_url"] = link
                    rp["product_group"] = _join_labels(prod["product_group"], rp["product_group"])
                    rp["product_group_description"] = _join_labels(prod["product_group_description"], rp["product_group_description"])
                    rp["product_group_url"] = link
                rexamine_products.extend(rex_prods)
            else:
                prod_details = extract_details_from_prod_page(webpage=webtxt, web_url=link)
                if len(prod) == 0:
                    continue

                prod["product_title"] = prod_details["product_title"]
                prod["product_short_description"] = prod_details["product_short_description"]
                prod["product_details_description"] = prod_details["product_details_description"]
                prod["product_table_details"] = prod_details["product_table_details"]
                products_with_info.append(prod)
        finished_products.extend(products_with_info)
        all_products = rexamine_products

    json.dump(finished_products, open("all_products.json", "w", encoding="utf-8"), ensure_ascii=False, sort_keys=True, indent=4)