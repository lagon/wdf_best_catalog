from openai import OpenAI
import os
import tqdm

import config.configuration as cfg

def cancel_all_batches():
    client = OpenAI(api_key=cfg.get_settings().open_ai_api_key)

    cancelled = []
    # pagination loop
    cursor = None
    while True:
        # fetch one page of batches
        print("Listing all batches...")
        response = client.batches.list(limit=100, after=cursor)
        batches = response.data

        if not batches:
            break

        for batch in tqdm.tqdm(batches):
            if batch.status in ("validating", "in_progress", "finalizing"):
                try:
                    client.batches.cancel(batch.id)
                    cancelled.append(batch.id)
                    print(f"Cancelled batch: {batch.id}")
                except Exception as e:
                    print(f"Failed to cancel {batch.id}: {e}")

        # move to next page
        cursor = response.last_id
        if not response.has_more:
            break

    if not cancelled:
        print("No running batches to cancel.")
    else:
        print(f"Total cancelled batches: {len(cancelled)}")


# def cancel_all_batches():
#     # List all batches
#     batches = client.batches.list(limit=100)
#
#     cancelled = []
#     for batch in tqdm.tqdm(batches.data):
#         if batch.status in ("validating", "in_progress", "finalizing"):
#             client.batches.cancel(batch.id)
#             cancelled.append(batch.id)
#             print(f"Cancelled batch: {batch.id}")
#
#     if not cancelled:
#         print("No running batches to cancel.")
#     else:
#         print(f"Total cancelled batches: {len(cancelled)}")

if __name__ == "__main__":
    cancel_all_batches()