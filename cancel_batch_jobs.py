import openai

client = openai.OpenAI()

# iterate through all pages
for batch in client.batches.list().data:  # list_batches is paginated
    if batch.status in {
        "validating",
        "enqueued",
        "running",
        "in_progress",
        "finalizing",
    }:
        client.batches.cancel(batch.id)
        print(f"Cancelled {batch.id}")
