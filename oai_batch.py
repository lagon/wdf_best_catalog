import abc
import collections as col
import datetime
import io
import json
import os
import pickle
import time
import typing as t

import openai as oai
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
import tqdm

import bucketize as buck

class WorkItem(abc.ABC):
    @abc.abstractmethod
    def get_id(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_jsonl_list(self) -> t.List[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def is_new(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def save_resposes(self, reponse_jsons: t.List[str]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def save_error_message(self, reponse_jsons: t.List[str]) -> None:
        raise NotImplementedError

class OAI_Worker(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def add_work_items(self, work_items: t.List[WorkItem]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def run_loop(self) -> None:
        raise NotImplementedError


class OAI_Direct(OAI_Worker):
    def __init__(self, client: oai.Client, working_dir: str, number_active_batches: int, max_work_items_to_add: int, tag: str):
        self.client = client
        self.working_dir = working_dir
        self.work_items: t.List = []
        self.number_active_batches = number_active_batches
        self._max_work_items_to_add = max_work_items_to_add
        self._tag = tag

    def add_work_items(self, work_items: t.List[WorkItem]) -> None:
        self.work_items.extend(work_items)

    def run_loop(self) -> None:
        for wi_num, work_item in enumerate(self.work_items):
            if work_item.is_new():
                lines = work_item.get_jsonl_list()
                all_responses = []
                for l in tqdm.tqdm(lines, desc=f"Working on WORK ITEM {wi_num+1}/{len(self.work_items)}", ncols=150):
                    wi_data = json.loads(l)
                    prompt = [
                        ChatCompletionSystemMessageParam(role="system", content=wi_data["body"]["messages"][0]["content"]),
                        ChatCompletionUserMessageParam(role="user", content=wi_data["body"]["messages"][1]["content"])
                    ]

                    response = self.client.chat.completions.create(model=wi_data["body"]["model"], messages=prompt, max_tokens=300)
                    resp_parsed = json.loads(response.to_json())
                    resp = {
                        "custom_id": wi_data["custom_id"],
                        "response": {
                            "body": {
                                "choices":[
                                    {"message": {"content": resp_parsed["choices"][0]["message"]["content"]}},
                                ]
                            }
                        }
                    }

                    all_responses.append(resp)

                work_item.save_resposes(all_responses)
        return



class OAI_Batch(OAI_Worker):
    def __init__(self, client: oai.Client, working_dir: str, number_active_batches: int, max_work_items_to_add: int, tag: str):
        self.client = client
        self.working_dir = working_dir
        self.work_items: t.List = []
        self.number_active_batches = number_active_batches
        self._max_work_items_to_add = max_work_items_to_add
        self._tag = tag

    def add_work_items(self, work_items: t.List[WorkItem]) -> None:
        self.work_items.extend(work_items)

    def _should_add_new_batch(self) -> bool:
        all_recorded_batches = buck.list_files_in_buckets(self.working_dir)
        all_recorded_batches = list(filter(lambda fn: os.path.basename(fn).startswith("llm_batch_info_") and fn.endswith(".running_batch.pickle"), all_recorded_batches))
        return len(all_recorded_batches) <= self.number_active_batches

    def _number_empty_work_item_slots(self) -> int:
        all_recorded_batches = buck.list_files_in_buckets(self.working_dir)
        all_recorded_batches = list(filter(lambda fn: os.path.basename(fn).startswith("llm_batch_info_") and fn.endswith(".running_batch.pickle"), all_recorded_batches))
        return max(0, self.number_active_batches - len(all_recorded_batches))
        # return len(all_recorded_batches) <= self.number_active_batches


    def _get_llm_batch_running_filename(self, batch_id: str, working_dir: str) -> str:
        buck.ensure_bucket_directory_exists(filename=f"llm_batch_info_{batch_id}.running_batch.pickle", buckets_root=working_dir)
        return buck.bucketized_filename(filename=f"llm_batch_info_{batch_id}.running_batch.pickle", buckets_root=working_dir)

    def _submit_new_batch(self, work_item: WorkItem) -> t.Optional[WorkItem]:
        jsonl_list = work_item.get_jsonl_list()
        if len(jsonl_list) == 0:
            return None

        jsonl_content = "\n".join(jsonl_list)
        oai_file = self.client.files.create(file=io.BytesIO(jsonl_content.encode("utf-8")), purpose="batch") #, expires_after={"anchor": "created_at", "seconds": 90000})
        batch_data = self.client.batches.create(
            input_file_id=oai_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            }
        )

        print("Submitted: {}".format(batch_data.id))
        with open(self._get_llm_batch_running_filename(batch_id=work_item.get_id(), working_dir=self.working_dir), "wb") as f:
            pickle.dump({
                "batch_id": batch_data.id,
                "batch_tag": self._tag,
                "work_item": work_item
            }, f)
        return work_item

    def batch_already_running(self, batch_id) -> bool:
        return os.path.isfile(self._get_llm_batch_running_filename(batch_id=batch_id, working_dir=self.working_dir))

    def _submit_next_available_work_item(self) -> None:
        while len(self.work_items) > 0:
            wi: WorkItem = self.work_items[0]
            self.work_items = self.work_items[1:]

            if (not self.batch_already_running(wi.get_id())) and wi.is_new():
                self._submit_new_batch(work_item=wi)
                break
        return

    def _check_all_batch_statuses(self) -> t.Tuple[t.List[t.Dict[str, t.Any]], int]:
        all_files = buck.list_files_in_buckets(self.working_dir)
        batch_filenames = list(filter(lambda f: f.endswith(".running_batch.pickle"), all_files))

        finished_out_files = []
        batch_status: col.Counter = col.Counter()

        not_started_batches = []
        started_batches = []

        for fn in batch_filenames:
            with open(os.path.join(self.working_dir, fn), "rb") as f:
                batch = pickle.load(f)

            if batch["batch_tag"] != self._tag:
                continue

            try:
                batch_info = self.client.batches.retrieve(batch_id=batch["batch_id"])
                batch_status.update([batch_info.status])

                if batch_info.status in ["completed"]:
                    finished_out_files.append({
                        "work_item": batch["work_item"],
                        "output_file_id": batch_info.output_file_id,
                        "error_file_id": batch_info.error_file_id
                    })
                    os.remove(fn)
                elif batch_info.status in ["failed", "expired", "cancelled", "cancelling"]:
                    self.add_work_items([batch["work_item"]])
                    os.remove(fn)
                elif batch_info.status in ["in_progress"]:
                    batch_age = datetime.datetime.now() - datetime.datetime.fromtimestamp(batch_info.created_at)
                    if batch_info.request_counts.completed == 0:
                        not_started_batches.append(f"Not started {batch_age} in queue")
                    else:
                        started_batches.append(f"{batch_info.request_counts.completed}/{batch_info.request_counts.total} ({batch_age})")
            except Exception as e:
                print(f"Batch {batch['batch_id']} is wrong. Error message follows: {e}")

        print("{}/{} batches being worked on. {}/{} not started. Progress: {}".format(len(started_batches), len(started_batches) + len(not_started_batches), len(not_started_batches), len(started_batches) + len(not_started_batches), " | ".join(started_batches)))

        print(batch_status)
        return finished_out_files, (batch_status.total() - len(finished_out_files))

    def _download_and_save_finished_batches(self, finished_file_ids: t.List[t.Dict[str, t.Any]]) -> None:
        for ffid in finished_file_ids:
            wi: WorkItem = ffid["work_item"]
            if ffid["output_file_id"] is not None:
                file_content = self.client.files.content(file_id=ffid["output_file_id"])
                response_lines = file_content.content.decode("utf-8").split("\n")
                reponse_jsons = []
                for line in response_lines:
                    if line != "":
                        reponse_jsons.append(json.loads(line))

                wi.save_resposes(reponse_jsons)

            else:
                file_content = self.client.files.content(file_id=ffid["error_file_id"])
                response_lines = file_content.content.decode("utf-8").split("\n")
                wi.save_error_message(response_lines)


    def run_loop(self) -> None:
        cnt_remaining_tasks: int = 0
        while (len(self.work_items) > 0) or (cnt_remaining_tasks > 0):
            try:
                print(f"Another iteration -- remaining work items: {len(self.work_items)}")
                something_happened: bool = False

                empty_slots = min([self._number_empty_work_item_slots(), self._max_work_items_to_add, len(self.work_items)])
                while empty_slots > 0:
                    self._submit_next_available_work_item()
                    something_happened = True
                    empty_slots -= 1

                finished_batches, cnt_remaining_tasks = self._check_all_batch_statuses()

                if len(finished_batches) > 0:
                    print("Some files ready to download")
                    self._download_and_save_finished_batches(finished_file_ids=finished_batches)
                    something_happened = True

                if not something_happened:
                    print("Waiting 2 minutes before another checks")
                    time.sleep(120)
            except oai.InternalServerError as e:
                print("Some weird trouble with OpenAI server")
                print(e)
                print("Waiting 2 minutes before another checks")
                time.sleep(120)
        return
