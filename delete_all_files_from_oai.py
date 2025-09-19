import os
import concurrent.futures as fut

import openai as oai
import tqdm

def delete_oai_file(file_id: str) -> None:
    cli.files.delete(file_id)

if __name__ == '__main__':
    cli = oai.Client(api_key=os.environ.get("OPEN_AI_API_KEY"))
    all_files = cli.files.list()

    exec: fut.ThreadPoolExecutor = fut.ThreadPoolExecutor(10)
    all_futures: list[fut.Future] = []

    for file in tqdm.tqdm(all_files, ncols=150, total=len(all_files.data), desc=f"Initiating removal job {len(all_files.data)}"):
        all_futures.append(exec.submit(delete_oai_file, file.id))

    for req in tqdm.tqdm(fut.as_completed(all_futures), desc="Delete all files", ncols=150, total=len(all_futures)):
        req.result()

