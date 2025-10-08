import abc
import collections
import os
import json
import hashlib
import typing as t

import openai as oai

import config.configuration as cfg
import oai_batch

class AbstractSelectionWorkItem(oai_batch.WorkItem):
    def __init__(self):
        super().__init__()
        self._item_id = ""

    def _set_id(self, item_id: str):
        self._item_id = item_id

    def get_id(self) -> str:
        return self._item_id

    def _get_request_body(self, request_id: str, prompt: str, input: str, model_name: str) -> str:
        body = {
            "custom_id": request_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "messages": [
                    {"role": "developer", "content": prompt},
                    {"role": "user", "content": input}
                ]
            }
        }
        return json.dumps(body, ensure_ascii=False)

    @abc.abstractmethod
    def _get_filename_prefix(self):
        raise NotImplementedError

    def _get_output_filename(self) -> str:
        return os.path.join(self._work_dir, f"{self._get_filename_prefix()}-{self.get_id()}.json")

    def is_new(self) -> bool:
        return not os.path.isfile(self._get_output_filename())

    def get_responses(self) -> t.Dict[str, str]:
        if not self.is_new():
            with open(self._get_output_filename(), "rt", encoding="utf-8") as f:
                return json.load(fp=f)
        else:
            raise Exception("Not yet evaluated")

    def save_error_message(self, reponse_jsons: t.List[str]) -> None:
        pass

class ExtendCustomerDescriptionSumWorkItem(AbstractSelectionWorkItem):
    def __init__(self, requested_product: str, additional_data: t.Dict[str, str], model_name: str, work_dir: str):
        super().__init__()
        self._requested_product = requested_product
        self._model_name = model_name
        self._work_dir = work_dir
        self._additional_data = additional_data
        os.makedirs(work_dir, exist_ok=True)

        self._set_id(hashlib.md5(";;;".join([requested_product, model_name]).encode()).hexdigest())

    def _get_filename_prefix(self):
        return "customer-extended"

    def get_jsonl_list(self) -> t.List[str]:
        extend_description_prompt: str = """Zakazník poptává produkt s následujícím popisem. Nahraď všechny zkratky. Dále zkus odhadnout a rozepsat, 
jak by takový produkt mohl vypadat, identifikuj typ a určení výrobku, požadovaný materiál, 
barvu a rozměry. Napiš jeden kratký odstavec s popisem. Odpovídej v češtině."""
#         identify_colour_prompt: str = """Zakazník poptává produkt s následujícím popisem. Pokus se identifikovat z popisu barvu poptavaného zboží.
# Pokud barva není zmíněna nebo se nedaří ji určit, odpověz: NENÍ. Vždy odpověz jen jedním slovem."""
        batch_input = [
            self._get_request_body(request_id="extend_description", model_name=self._model_name, prompt=extend_description_prompt, input=self._requested_product)
        ]

        return batch_input

    def save_resposes(self, reponse_jsons: t.List[str]) -> None:
        store_object = {}

        for resp in reponse_jsons:
            custom_id = resp["custom_id"]
            data = resp["response"]["body"]["choices"][0]["message"]["content"]
            store_object[custom_id] = data
        store_object["additional_data"] = self._additional_data

        self._response_data = store_object

        with open(self._get_output_filename(), "wt", encoding="utf-8") as f:
            json.dump(
                obj=store_object,
                fp=f,
                indent=4,
                ensure_ascii=False
            )

class SingleProductTypeQueryWorkItem(AbstractSelectionWorkItem):
    def __init__(self, customer_request_description: str, product_types_description: t.Dict[str, str], additional_data: t.Dict[str, str], model_name: str, work_dir: str):
        super().__init__()
        self._customer_request_description = customer_request_description
        self._product_types_description = product_types_description
        self._model_name = model_name
        self._work_dir = work_dir
        self._additional_data = additional_data
        os.makedirs(work_dir, exist_ok=True)

        add_dta = "@@@".join([f"{k}={v}" for k, v in additional_data.items()])

        self._set_id(hashlib.md5(";;;".join([customer_request_description, "|||".join(product_types_description), add_dta, model_name]).encode()).hexdigest())

    def _get_filename_prefix(self):
        return "product-type"

    def get_jsonl_list(self) -> t.List[str]:
        extend_description_prompt: str = """You are selling products sold by manufacturer of prefabricated concrete slabs, tiles and other 
similar products and accessories. Your task is to estimate how well a customer requested product fit into the product type. Each product
type consists of number of products with similar characteristics and use.

Requested product is described in two ways - directly described by a customer (in CUSTOMER DESCRIPTION field) and
by an expanded version as guessed by LLM (in EXTENDED DESCRIPTION field). Customer request is:
{CUSTOMER_REQUEST}

####################################

The product type is described by it's name, in PRODUCT TYPE field, product type's description, in PRODUCT TYPE DESCRIPTION field.
In addition there are two lists - list of all product names, in case they are fully or partially mentioned in request, in 
ALL PRODUCT NAMES field. The second list contains colour variations of products within the product type, in AVAILABLE COLORS field.
If CUSTOMER DESCRIPTION contains full or partial name of any product mentioned in ALL PRODUCT NAMES field, it's a big boost for 
the fit score, equals to 1, and indication that a customer needs exactly this product type.

Product type fields:
{PRODUCT_TYPE}

Respond with a single number indicating how well the customer requested product fit into the product type. The scale is 0 to 1. 
0 means no fit, customer want something else, while 1 indicates strong fit."""
        batch_input = []
        for prod_type, prod_type_desc in self._product_types_description.items():
            instructions = extend_description_prompt.format(CUSTOMER_REQUEST=self._customer_request_description, PRODUCT_TYPE=prod_type_desc)
            self._additional_data["PRODUCT TYPE INSTRUCTIONS"] = instructions
            batch_input.append(
                self._get_request_body(request_id=prod_type, model_name=self._model_name, prompt=instructions, input="What is the fit between customer request and product type? Number only.")
            )
        return batch_input

    def save_resposes(self, reponse_jsons: t.List[str]) -> None:
        store_object = {}

        for resp in reponse_jsons:
            custom_id = resp["custom_id"]
            data = resp["response"]["body"]["choices"][0]["message"]["content"]
            store_object[custom_id] = data
        store_object["additional_data"] = self._additional_data

        self._response_data = store_object

        with open(self._get_output_filename(), "wt", encoding="utf-8") as f:
            json.dump(
                obj=store_object,
                fp=f,
                indent=4,
                ensure_ascii=False
            )

class SingleProductGroupQueryWorkItem(AbstractSelectionWorkItem):
    def __init__(self, customer_request_description: str, product_group_data: t.Dict[str, str], additional_data: t.Dict[str, t.Union[str,float]], model_name: str, work_dir: str):
        super().__init__()
        self._pg_description = product_group_data["AH PRODUCT GROUP DESCRIPTION"]
        self._pg_prod_list = product_group_data["AH PRODUCT"]
        self._pg_prod_colours = product_group_data["AH PRODUCT ALL COLOURS"]
        self._pg_name = product_group_data["AH PRODUCT GROUP"]
        self._customer_request_description = customer_request_description
        self._additional_data = additional_data
        self._model_name = model_name
        self._work_dir = work_dir

        id_str = "|||".join([self._customer_request_description, self._pg_description, ";;;".join(self._pg_prod_list), ";;;".join(self._pg_prod_colours), self._model_name])
        self._set_id(hashlib.md5(id_str.encode()).hexdigest())

    def _get_filename_prefix(self):
        return "product-group"

    def get_jsonl_list(self) -> t.List[str]:
        extend_description_prompt: str = """You are selling products sold by manufacturer of prefabricated concrete slabs, tiles and other 
similar products and accessories. Your task is to estimate how well a customer requested product fit into the product group. Each product
group consists of number of products with similar characteristics and use.

Requested product is described in two ways - directly described by a customer (in CUSTOMER DESCRIPTION field) and
by an expanded version as guessed by LLM (in EXTENDED DESCRIPTION field). Customer request is:
{CUSTOMER_REQUEST}

####################################

The product group is described by it's name, in PRODUCT GROUP field, product group's description, in PRODUCT GROUP DESCRIPTION field.
In addition there are two lists - list of all product names, in case they are fully or partially mentioned in request, in 
ALL PRODUCT NAMES field. The second list contains colour variations of products within the product type, in AVAILABLE COLORS field.
If CUSTOMER DESCRIPTION contains full or partial name of any product mentioned in ALL PRODUCT NAMES field, it's a big boost for 
the fit score, equals to 1, and indication that a customer needs exactly this product group.

Product type fields:
{PRODUCT_GROUP}

Respond with a single number indicating how well the customer requested product fit into the product type. The scale is 0 to 1. 
0 means no fit, customer want something else, while 1 indicates strong fit."""

        prod_grp_str = f"""
PRODUCT GROUP: '{self._pg_name}'
PRODUCT GROUP DESCRIPTION: '{self._pg_description}'
ALL PRODUCT NAMES: '{", ".join(self._pg_prod_list)}'
AVAILABLE COLORS: '{", ".join(self._pg_prod_colours)}')}}'
"""

        batch_input = []
        instructions = extend_description_prompt.format(CUSTOMER_REQUEST=self._customer_request_description, PRODUCT_GROUP=prod_grp_str)
        self._additional_data["PRODUCT GROUP INSTRUCTIONS"] = instructions
        batch_input.append(
            self._get_request_body(request_id=self._pg_name, model_name=self._model_name, prompt=instructions, input="What is the fit between customer request and product type? Number only.")
        )
        return batch_input

    def save_resposes(self, reponse_jsons: t.List[str]) -> None:
        store_object = {}

        for resp in reponse_jsons:
            custom_id = resp["custom_id"]
            data = resp["response"]["body"]["choices"][0]["message"]["content"]
            store_object[custom_id] = data
        store_object["additional_data"] = self._additional_data

        self._response_data = store_object

        with open(self._get_output_filename(), "wt", encoding="utf-8") as f:
            json.dump(
                obj=store_object,
                fp=f,
                indent=4,
                ensure_ascii=False
            )

class SingleProductQueryWorkItem(AbstractSelectionWorkItem):
    def __init__(self, customer_request_description: str, product_data: t.Dict[str, str], additional_data: t.Dict[str, t.Union[str,float]], all_prod_colours: t.List[str], model_name: str, work_dir: str):
        super().__init__()
        self._customer_request_description = customer_request_description
        self._prod_name = product_data["product_title"]
        self._prod_summary = product_data["product_summary"]
        self._prod_param_colour = product_data["product_param_colour"]
        self._prod_dimension_table = product_data["product_table_details"]
        self._prod_param_thickness = product_data["product_param_height"]
        self._prod_parameters = product_data["product_parameters"]
        self._additional_data = additional_data
        self._alt_colours = all_prod_colours.copy()
        if self._prod_param_colour in self._alt_colours:
            self._alt_colours.remove(self._prod_param_colour)

        self._model_name = model_name
        self._work_dir = work_dir

        id_str = "|||".join([self._prod_name, self._prod_summary, self._model_name])
        self._set_id(hashlib.md5(id_str.encode()).hexdigest())

    def _get_filename_prefix(self):
        return "_product"

    def get_jsonl_list(self) -> t.List[str]:
        extend_description_prompt: str = """You are selling products sold by manufacturer of prefabricated concrete slabs, tiles and other 
similar products and accessories. Your task is to estimate how well a customer requested product fits the particular product. Products
differ mainly in colour and possibly dimensions (or height of tiles, defining load bearing capacity - walkable, light-traffic or heavy traffic).

Requested product is described in two ways - directly described by a customer (in CUSTOMER DESCRIPTION field) and by an expanded version 
as guessed by LLM (in EXTENDED DESCRIPTION field). Customer request is:
    {CUSTOMER_REQUEST}
    
####################################
    
The product is described by it's name, in PRODUCT field, product description, in PRODUCT SUMMARY field, and then by it's color and then by it's dimensions.
Colour is in PRODUCT COLOUR field. Dimensions are represented in two ways - by PRODUCT DIMENSIONS field, which is json-like dictionary containing product label
and then width, length and height in millimeters. Second way is by PRODUCT THICKNESS indicating product thickness, if available, otherwise 'NA'. In 
PRODUCT PARAMETERS are all optional parameters that were collected from the website. The format is json-like string with keys indicating meaning of the value.

If customer request contains full or partial name of a colour, it's a strong signal, he wants exactly this product group. In terms of dimensions, they
can be much more fuzzy, but closer the better. If customer requested product colour matches colours of alternative products (listed 
in ALTERNATIVE COLOURS field), the fit of this particulat product goes significantly down.
    
Product type fields:
{PRODUCT_FIELD}

ALTERNATIVE COLOURS: {ALTERNATIVE_COLOURS}

Respond with a single number indicating how well the customer requested product fit into the product type. The scale is 0 to 1. 0 means no fit, customer
want something else, while 1 indicates strong fit."""

        prod_str = f"""
    PRODUCT: '{self._prod_name}'
    PRODUCT SUMMARY: '{self._prod_summary}'
    PRODUCT COLOUR: '{self._prod_param_colour}'
    PRODUCT DIMENSIONS: '{self._prod_dimension_table}'
    PRODUCT THICKNESS: '{self._prod_param_thickness}'
    PRODUCT PARAMETERS: '{self._prod_parameters}'
    """

        batch_input = []
        instructions = extend_description_prompt.format(CUSTOMER_REQUEST=self._customer_request_description, PRODUCT_FIELD=prod_str, ALTERNATIVE_COLOURS=",".join(self._alt_colours))
        self._additional_data["PRODUCT INSTRUCTIONS"] = instructions
        batch_input.append(
            self._get_request_body(request_id=self._prod_name, model_name=self._model_name, prompt=instructions, input="What is the fit between customer request and the product? Number only.")
        )
        return batch_input

    def save_resposes(self, reponse_jsons: t.List[str]) -> None:
        store_object = {}

        for resp in reponse_jsons:
            custom_id = resp["custom_id"]
            data = resp["response"]["body"]["choices"][0]["message"]["content"]
            store_object[custom_id] = data
        store_object["additional_data"] = self._additional_data

        self._response_data = store_object

        with open(self._get_output_filename(), "wt", encoding="utf-8") as f:
            json.dump(
                obj=store_object,
                fp=f,
                indent=4,
                ensure_ascii=False
            )



def expand_customer_requested_product(request_offer_list: t.List[t.Dict], batch: oai_batch.OAI_Batch, desc_expansion_model_name: str, work_dir: str) -> t.List[t.Dict]:
    desc_expansion_work_items: t.List[ExtendCustomerDescriptionSumWorkItem] = []

    for req_off in request_offer_list:
        additional_data = {
            "CUSTOMER DESCRIPTION": req_off["CUSTOMER DESCRIPTION"],
            "OFFERED PRODUCT NAME": req_off["OFFERED PRODUCT NAME"]
        }
        desc_expansion_work_items.append(
            ExtendCustomerDescriptionSumWorkItem(requested_product=req_off["CUSTOMER DESCRIPTION"], additional_data=additional_data, model_name=desc_expansion_model_name, work_dir=work_dir)
        )

    batch.add_work_items(desc_expansion_work_items)
    batch.run_loop()

    output = []
    for work_item in desc_expansion_work_items:
        wi = work_item.get_responses()
        add_data = {
            "FINAL OUTPUT": wi["additional_data"]["OFFERED PRODUCT NAME"],
            "CUSTOMER DESC": wi["additional_data"]["CUSTOMER DESCRIPTION"],
            "EXTENDED DESCRIPTION": wi["extend_description"]
        }
        output.append(add_data)
    return output

def find_product_type(desc_expansion: t.List[t.Dict[str, str]], alt_hierarchy_db: t.Dict[str, t.Dict[str, t.Any]], batch: oai_batch.OAI_Batch, desc_expansion_model_name: str, work_dir: str) -> t.Dict[str, t.List[t.Dict[str, str]]]:

    all_product_types: t.Dict[str, str] = {}
    for prod_types in alt_hierarchy_db["product_types_db"].values():
        all_product_types[prod_types["AH PRODUCT TYPE NAME"]] = f"""
PRODUCT TYPE: {prod_types["AH PRODUCT TYPE NAME"]}
PRODUCT TYPE DESCRIPTION: {prod_types["AH PRODUCT TYPE DESCRIPTION"]}
AVAILABLE COLORS: {prod_types["AH PRODUCT TYPE COLOURS"]}
ALL PRODUCT NAMES: {prod_types["AH PRODUCT GROUPS"]}
"""

    all_work_items: t.List[SingleProductTypeQueryWorkItem] = []
    for desc in desc_expansion:
        req_prod = f"""
CUSTOMER DESCRIPTION: '{desc["CUSTOMER DESC"]}'
EXTENDED DESCRIPTION: '{desc["EXTENDED DESCRIPTION"]}' 
        """
        wi = SingleProductTypeQueryWorkItem(customer_request_description=req_prod, product_types_description=all_product_types, additional_data=desc, model_name=desc_expansion_model_name, work_dir=work_dir)
        all_work_items.append(wi)

    batch.add_work_items(all_work_items)
    batch.run_loop()

    output = []
    no_match = []
    for wi in all_work_items:
        resp = wi.get_responses()
        add_data = resp["additional_data"]
        for prod_type, prot_type_val in resp.items():
            if prod_type == "additional_data":
                continue
            if prod_type not in set(all_product_types.keys()):
                print(f"UNKNOWN PRODUCT TYPE: {prod_type}")
            else:
                prot_type_prob = float(prot_type_val)
                if prot_type_prob <= 0.3:
                    no_match.append(resp)
                else:
                    output.append({
                        "FINAL OUTPUT": add_data["FINAL OUTPUT"],
                        "CUSTOMER DESC": add_data["CUSTOMER DESC"],
                        "EXTENDED DESCRIPTION": add_data["EXTENDED DESCRIPTION"],
                        "PRODUCT TYPE": prod_type,
                        "PRODUCT TYPE PCT": prot_type_prob,
                        "PRODUCT TYPE INSTRUCTIONS": add_data["PRODUCT TYPE INSTRUCTIONS"],
                    })

    return {
        "match": output,
        "no_match": no_match
    }

def find_product_group(selected_product_type: t.List[t.Dict[str, str]], alt_hierarchy_db: t.Dict[str, t.Dict[str, t.Any]], batch: oai_batch.OAI_Batch, desc_expansion_model_name: str, work_dir: str) -> t.Dict[str, t.List[t.Dict[str, str]]]:
    all_work_items: t.List[SingleProductGroupQueryWorkItem] = []
    for spt in selected_product_type:
        all_prod_groups = alt_hierarchy_db["product_types_db"][spt["PRODUCT TYPE"]]["AH PRODUCT GROUPS"]
        for pg in all_prod_groups:
            req_prod = f"""
CUSTOMER DESCRIPTION: '{spt["CUSTOMER DESC"]}'
EXTENDED DESCRIPTION: '{spt["EXTENDED DESCRIPTION"]}' 
        """
            prod_group_details = alt_hierarchy_db["product_group_db"][pg]
            all_work_items.append(SingleProductGroupQueryWorkItem(customer_request_description=req_prod, product_group_data=prod_group_details, additional_data=spt.copy(), model_name=desc_expansion_model_name, work_dir=work_dir))

    batch.add_work_items(all_work_items)
    batch.run_loop()

    output = []
    no_match = []

    for wi in all_work_items:
        resp = wi.get_responses()
        add_data = resp["additional_data"]
        for prod_group, prod_group_val in resp.items():
            if prod_group == "additional_data":
                continue
            if prod_group not in alt_hierarchy_db["product_group_db"]:
                print(f"UNKNOWN PRODUCT TYPE: {prod_group}")
            else:
                prod_group_prob = float(prod_group_val)
                if prod_group_prob <= 0.4:
                    no_match.append(resp)
                else:
                    output.append({
                        "FINAL OUTPUT": add_data["FINAL OUTPUT"],
                        "CUSTOMER DESC": add_data["CUSTOMER DESC"],
                        "EXTENDED DESCRIPTION": add_data["EXTENDED DESCRIPTION"],
                        "PRODUCT TYPE": add_data["PRODUCT TYPE"],
                        "PRODUCT TYPE PCT": add_data["PRODUCT TYPE PCT"],
                        "PRODUCT TYPE INSTRUCTIONS": add_data["PRODUCT TYPE INSTRUCTIONS"],
                        "PRODUCT GROUP": prod_group,
                        "PRODUCT GROUP PCT": prod_group_prob,
                        "PRODUCT GROUP INSTRUCTIONS": add_data["PRODUCT GROUP INSTRUCTIONS"],
                    })
    return {
        "match": output,
        "no_match": no_match
    }

def find_products(selected_product_groups: t.List[t.Dict[str, t.Union[str, float]]], alt_hierarchy_db: t.Dict[str, t.Any], batch: oai_batch.OAI_Batch, desc_expansion_model_name: str, work_dir: str) -> t.Dict[str, t.List[t.Dict[str, t.Union[str, float]]]]:
    all_work_items: t.List[SingleProductQueryWorkItem] = []
    for spt in selected_product_groups:
        products = alt_hierarchy_db["product_group_db"][spt["PRODUCT GROUP"]]["AH PRODUCT"]
        req_prod = f"""
CUSTOMER DESCRIPTION: '{spt["CUSTOMER DESC"]}'
EXTENDED DESCRIPTION: '{spt["EXTENDED DESCRIPTION"]}' 
        """

        for prod in products:
            all_work_items.append(
                SingleProductQueryWorkItem(
                    customer_request_description=req_prod,
                    product_data=alt_hierarchy_db["product_db"][prod],
                    all_prod_colours=alt_hierarchy_db["product_group_db"][spt["PRODUCT GROUP"]]["AH PRODUCT ALL COLOURS"],
                    additional_data=spt.copy(),
                    model_name=desc_expansion_model_name,
                    work_dir=work_dir
                )
            )
    batch.add_work_items(all_work_items)
    batch.run_loop()

    output = []
    no_match = []
    for wi in all_work_items:
        resp = wi.get_responses()
        add_data = resp["additional_data"]
        for prod, prod_val in resp.items():
            if prod == 'BEST - BEATON DRENO PŘÍRODNÍ':
                print("XXXX")
            if prod == "additional_data":
                continue
            if prod not in alt_hierarchy_db["product_db"]:
                print(f"UNKNOWN PRODUCT TYPE: {prod}")
            else:
                prod_prob = float(prod_val)
                if prod_prob <= 0.5:
                    no_match.append(resp)
                else:
                    output.append({
                        "FINAL OUTPUT": add_data["FINAL OUTPUT"],
                        "CUSTOMER DESC": add_data["CUSTOMER DESC"],
                        "EXTENDED DESCRIPTION": add_data["EXTENDED DESCRIPTION"],
                        "PRODUCT TYPE": add_data["PRODUCT TYPE"],
                        "PRODUCT TYPE PCT": add_data["PRODUCT TYPE PCT"],
                        "PRODUCT TYPE INSTRUCTIONS": add_data["PRODUCT TYPE INSTRUCTIONS"],
                        "PRODUCT GROUP": add_data["PRODUCT GROUP"],
                        "PRODUCT GROUP PCT": add_data["PRODUCT GROUP PCT"],
                        "PRODUCT GROUP INSTRUCTIONS": add_data["PRODUCT GROUP INSTRUCTIONS"],
                        "PRODUCT": prod,
                        "PRODUCT PCT": prod_prob,
                        "PRODUCT INSTRUCTIONS": add_data["PRODUCT INSTRUCTIONS"],
                    })
    return {
        "match": output,
        "no_match": no_match
    }

def main():
    with open(os.path.join(cfg.get_settings().main_results_path_dir, "alternative_hierarchy_db.json"), 'rt', encoding='utf-8') as f:
        alt_hierarchy_db = json.load(f)

    request_offer_list = [
        # {
        #     "CUSTOMER DESCRIPTION": "DLAŽBY VEGETAČNÍ Z TVÁRNIC Z PLASTICKÝCH HMOT -  dlaždice tl. 60mm",
        #     "OFFERED PRODUCT NAME": ""
        # },
        {
            "CUSTOMER DESCRIPTION": "Beleza, 100m2, Moka",
            "OFFERED PRODUCT NAME": ""
        },
        # {
        #     "CUSTOMER DESCRIPTION": "DRENÁŽNÍ VÝUSŤ Z BETON DÍLCŮ - pro drenáž DN 150mm",
        #     "OFFERED PRODUCT NAME": ""
        # },
        # {
        #     "CUSTOMER DESCRIPTION": "DRENÁŽNÍ VÝUSŤ Z BETON DÍLCŮ - pro drenáž DN 200mm",
        #     "OFFERED PRODUCT NAME": ""
        # },
        # {
        #     "CUSTOMER DESCRIPTION": "KRYTY Z BETON DLAŽDIC SE ZÁMKEM BAREV RELIÉF TL 60MM",
        #     "OFFERED PRODUCT NAME": "BEST - BEATON PRO NEVIDOMÉ ČERVENÁ"
        # },
        # {
        #     "CUSTOMER DESCRIPTION": "KRYTY Z BETON DLAŽDIC SE ZÁMKEM BAREV TL 60MM",
        #     "OFFERED PRODUCT NAME": "BEST - BEATON ČERVENÁ"
        # },
        # {
        #     "CUSTOMER DESCRIPTION": "KRYTY Z BETON DLAŽDIC SE ZÁMKEM ŠEDÝCH TL 60MM",
        #     "OFFERED PRODUCT NAME": "BEST - BEATON PŘÍRODNÍ"
        # },
        # {
        #     "CUSTOMER DESCRIPTION": "NÁSTUPIŠTNÍ OBRUBNÍKY BETONOVÉ - bezbariérový obrubník HK 400/290/1000-P",
        #     "OFFERED PRODUCT NAME": "BEST - ZASTÁVKOVÝ OBRUBNÍK PŘÍMÝ PŘÍRODNÍ"
        # },
        # {
        #     "CUSTOMER DESCRIPTION": "PŘÍKOPOVÉ ŽLABY Z BETON SVAHOVÝCH TVÁRNIC ŠÍŘ 600MM - KASKÁDOVITÝ SKLUZ",
        #     "OFFERED PRODUCT NAME": "BEST - ŽLAB I PŘÍRODNÍ"
        # },
        # {
        #     "CUSTOMER DESCRIPTION": "PŘÍKOPOVÉ ŽLABY Z BETON TVÁRNIC ŠÍŘ 600MM",
        #     "OFFERED PRODUCT NAME": "BEST - ŽLAB I PŘÍRODNÍ"
        # },
        # {
        #     "CUSTOMER DESCRIPTION": "SILNIČNÍ A CHODNÍKOVÉ OBRUBY Z BETONOVÝCH OBRUBNÍKŮ ŠÍŘ 100MM - výška 250mm",
        #     "OFFERED PRODUCT NAME": "BEST - SINIA I PŘÍRODNÍ"
        # },
        # {
        #     "CUSTOMER DESCRIPTION": "SILNIČNÍ A CHODNÍKOVÉ OBRUBY Z BETONOVÝCH OBRUBNÍKŮ ŠÍŘ 150MM - výška 150mm (nájezdový)",
        #     "OFFERED PRODUCT NAME": "BEST - MONO NÁJEZDOVÝ ROVNÝ PŘÍRODNÍ"
        # },
        # {
        #     "CUSTOMER DESCRIPTION": "SILNIČNÍ A CHODNÍKOVÉ OBRUBY Z BETONOVÝCH OBRUBNÍKŮ ŠÍŘ 150MM - výška 250mm",
        #     "OFFERED PRODUCT NAME": "BEST - MONO II PŘÍRODNÍ"
        # },
        # {
        #     "CUSTOMER DESCRIPTION": "SILNIČNÍ A CHODNÍKOVÉ OBRUBY Z BETONOVÝCH OBRUBNÍKŮ ŠÍŘ 150MM - výška 300mm",
        #     "OFFERED PRODUCT NAME": "BEST - MONO I PŘÍRODNÍ"
        # },
        # {
        #     "CUSTOMER DESCRIPTION": "ZÁHONOVÉ OBRUBY Z BETONOVÝCH OBRUBNÍKŮ ŠÍŘ 50MM - výška 200mm",
        #     "OFFERED PRODUCT NAME": "BEST - PARKAN II PŘÍRODNÍ"
        # },
        # {
        #     "CUSTOMER DESCRIPTION": "ŽLABY A RIGOLY DLÁŽDĚNÉ Z BETONOVÝCH DLAŽDIC - 500x250x80mm",
        #     "OFFERED PRODUCT NAME": "BEST - NAVIGA PŘÍRODNÍ"
        # }
    ]

    all_outputs_list = []
    all_no_match_list = []

    client: oai.OpenAI = oai.OpenAI(api_key=cfg.get_settings().open_ai_api_key)
    batch: oai_batch.OAI_Batch = oai_batch.OAI_Batch(client=client, number_active_batches=100, max_work_items_to_add=100, return_work_items=False, working_dir=cfg.get_settings().product_resolution_cache_dir)

    desc_expansion: t.List[t.Dict] = expand_customer_requested_product(request_offer_list=request_offer_list, batch=batch, desc_expansion_model_name=cfg.get_settings().customer_description_expansion_model_name, work_dir=cfg.get_settings().product_resolution_cache_dir)
    output = find_product_type(desc_expansion=desc_expansion, alt_hierarchy_db=alt_hierarchy_db, batch=batch, desc_expansion_model_name=cfg.get_settings().hierarchy_inference_model_name, work_dir=cfg.get_settings().product_resolution_cache_dir)
    selected_product_types = output["match"]
    all_no_match_list.extend(output["no_match"])

    output = find_product_group(selected_product_type=selected_product_types, alt_hierarchy_db=alt_hierarchy_db, batch=batch, desc_expansion_model_name=cfg.get_settings().hierarchy_inference_model_name, work_dir=cfg.get_settings().product_resolution_cache_dir)
    selected_product_groups = output["match"]
    all_no_match_list.extend(output["no_match"])

    output = find_products(selected_product_groups=selected_product_groups, alt_hierarchy_db=alt_hierarchy_db, batch=batch, desc_expansion_model_name=cfg.get_settings().hierarchy_inference_model_name, work_dir=cfg.get_settings().product_resolution_cache_dir)
    selected_products = output["match"]
    all_no_match_list.extend(output["no_match"])

    cust_reqests_resolved: t.Dict[str, t.List[t.Dict]] = {}
    for sp in selected_products:
        if sp["CUSTOMER DESC"] not in cust_reqests_resolved:
            cust_reqests_resolved[sp["CUSTOMER DESC"]] = []

        cust_reqests_resolved[sp["CUSTOMER DESC"]].append(sp)

    for cust_req in cust_reqests_resolved.keys():
        cust_reqests_resolved[cust_req] = sorted(cust_reqests_resolved[cust_req], key=lambda k: k["PRODUCT PCT"], reverse=True)

    with open(os.path.join(cfg.get_settings().main_results_path_dir, "resolved_products.json"), "wt", encoding="utf-8") as f:
        json.dump(obj=cust_reqests_resolved, fp=f, indent=4, ensure_ascii=False)
    with open(os.path.join(cfg.get_settings().main_results_path_dir, "failed_resolved_products.json"), "wt", encoding="utf-8") as f:
        json.dump(obj=all_no_match_list, fp=f, indent=4, ensure_ascii=False)

    print(cust_reqests_resolved)




if __name__ == '__main__':
    main()