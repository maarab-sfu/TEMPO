from huggingface_hub import HfApi

REPO_ID = "chengyenhsieh/TAO-Amodal"

api = HfApi()

files = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type="dataset",
)

print(f"Total repository files: {len(files)}")

print("\nFiles potentially containing Charades validation frames:")

for filename in files:
    name = filename.lower()

    if "charades" in name and (
        "val" in name or "validation" in name
    ):
        print(filename)