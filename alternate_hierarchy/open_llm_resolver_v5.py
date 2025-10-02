import collections
import os
import json
import hashlib
import typing as t

import openai as oai

import config.configuration as cfg
import oai_batch

class ExtendCustomerDescriptionSumWorkItem(oai_batch.WorkItem):
    def __init__(self, requested_product: str, additional_data: t.Dict[str, str], model_name: str, work_dir: str):
        super().__init__()
        self._requested_product = requested_product
        self._model_name = model_name
        self._work_dir = work_dir
        self._additional_data = additional_data
        os.makedirs(work_dir, exist_ok=True)

        self._item_id = hashlib.md5(";;;".join([requested_product, model_name]).encode()).hexdigest()

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

    def _get_output_filename(self) -> str:
        return os.path.join(self._work_dir, f"customer-extended-{self.get_id()}.json")

    def is_new(self) -> bool:
        return not os.path.isfile(self._get_output_filename())

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

    def get_responses(self) -> t.Dict[str, str]:
        if not self.is_new():
            with open(self._get_output_filename(), "rt", encoding="utf-8") as f:
                return json.load(fp=f)
        else:
            raise Exception("Not yet evaluated")

    def save_error_message(self, reponse_jsons: t.List[str]) -> None:
        pass


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
            "FINAL OUTPUT": wi["additional_data"]["FINAL OUTPUT"],
            "CUSTOMER DESC": wi["additional_data"]["CUSTOMER DESC"],
            "EXTENDED DESCRIPTION": wi["extend_description"]
        }
        output.append(add_data)
    return output


def main():
    with open(os.path.join(cfg.get_settings().main_results_path_dir, "alternative_hierarchy_db.json"), 'rt', encoding='utf-8') as f:
        alt_hierarchy_db = json.load(f)

    request_offer_list = [
        {
            "CUSTOMER DESCRIPTION": "DLAŽBY VEGETAČNÍ Z TVÁRNIC Z PLASTICKÝCH HMOT -  dlaždice tl. 60mm",
            "OFFERED PRODUCT NAME": ""
        },
        {
            "CUSTOMER DESCRIPTION": "DRENÁŽNÍ VÝUSŤ Z BETON DÍLCŮ - pro drenáž DN 150mm",
            "OFFERED PRODUCT NAME": ""
        },
        {
            "CUSTOMER DESCRIPTION": "DRENÁŽNÍ VÝUSŤ Z BETON DÍLCŮ - pro drenáž DN 200mm",
            "OFFERED PRODUCT NAME": ""
        },
        {
            "CUSTOMER DESCRIPTION": "KRYTY Z BETON DLAŽDIC SE ZÁMKEM BAREV RELIÉF TL 60MM",
            "OFFERED PRODUCT NAME": "BEST - BEATON PRO NEVIDOMÉ ČERVENÁ"
        },
        {
            "CUSTOMER DESCRIPTION": "KRYTY Z BETON DLAŽDIC SE ZÁMKEM BAREV TL 60MM",
            "OFFERED PRODUCT NAME": "BEST - BEATON ČERVENÁ"
        },
        {
            "CUSTOMER DESCRIPTION": "KRYTY Z BETON DLAŽDIC SE ZÁMKEM ŠEDÝCH TL 60MM",
            "OFFERED PRODUCT NAME": "BEST - BEATON PŘÍRODNÍ"
        },
        {
            "CUSTOMER DESCRIPTION": "NÁSTUPIŠTNÍ OBRUBNÍKY BETONOVÉ - bezbariérový obrubník HK 400/290/1000-P",
            "OFFERED PRODUCT NAME": "BEST - ZASTÁVKOVÝ OBRUBNÍK PŘÍMÝ PŘÍRODNÍ"
        },
        {
            "CUSTOMER DESCRIPTION": "PŘÍKOPOVÉ ŽLABY Z BETON SVAHOVÝCH TVÁRNIC ŠÍŘ 600MM - KASKÁDOVITÝ SKLUZ",
            "OFFERED PRODUCT NAME": "BEST - ŽLAB I PŘÍRODNÍ"
        },
        {
            "CUSTOMER DESCRIPTION": "PŘÍKOPOVÉ ŽLABY Z BETON TVÁRNIC ŠÍŘ 600MM",
            "OFFERED PRODUCT NAME": "BEST - ŽLAB I PŘÍRODNÍ"
        },
        {
            "CUSTOMER DESCRIPTION": "SILNIČNÍ A CHODNÍKOVÉ OBRUBY Z BETONOVÝCH OBRUBNÍKŮ ŠÍŘ 100MM - výška 250mm",
            "OFFERED PRODUCT NAME": "BEST - SINIA I PŘÍRODNÍ"
        },
        {
            "CUSTOMER DESCRIPTION": "SILNIČNÍ A CHODNÍKOVÉ OBRUBY Z BETONOVÝCH OBRUBNÍKŮ ŠÍŘ 150MM - výška 150mm (nájezdový)",
            "OFFERED PRODUCT NAME": "BEST - MONO NÁJEZDOVÝ ROVNÝ PŘÍRODNÍ"
        },
        {
            "CUSTOMER DESCRIPTION": "SILNIČNÍ A CHODNÍKOVÉ OBRUBY Z BETONOVÝCH OBRUBNÍKŮ ŠÍŘ 150MM - výška 250mm",
            "OFFERED PRODUCT NAME": "BEST - MONO II PŘÍRODNÍ"
        },
        {
            "CUSTOMER DESCRIPTION": "SILNIČNÍ A CHODNÍKOVÉ OBRUBY Z BETONOVÝCH OBRUBNÍKŮ ŠÍŘ 150MM - výška 300mm",
            "OFFERED PRODUCT NAME": "BEST - MONO I PŘÍRODNÍ"
        },
        {
            "CUSTOMER DESCRIPTION": "ZÁHONOVÉ OBRUBY Z BETONOVÝCH OBRUBNÍKŮ ŠÍŘ 50MM - výška 200mm",
            "OFFERED PRODUCT NAME": "BEST - PARKAN II PŘÍRODNÍ"
        },
        {
            "CUSTOMER DESCRIPTION": "ŽLABY A RIGOLY DLÁŽDĚNÉ Z BETONOVÝCH DLAŽDIC - 500x250x80mm",
            "OFFERED PRODUCT NAME": "BEST - NAVIGA PŘÍRODNÍ"
        }
    ]

    all_outputs_list = []
    client: oai.OpenAI = oai.OpenAI(api_key=cfg.get_settings().open_ai_api_key)
    oai_batch.OAI_Batch(client=client, number_active_batches=1, max_work_items_to_add=1, return_work_items=False, working_dir=cfg.get_settings().product_desc_expansion_request_cache_dir)

    desc_expansion: t.List[t.Dict] = expand_customer_requested_product(request_offer_list=request_offer_list, batch=batch, desc_expansion_model_name=desc_expansion_model_name, work_dir=work_dir)
    find_product_type(desc_expansion=desc_expansion, )


if __name__ == '__main__':
