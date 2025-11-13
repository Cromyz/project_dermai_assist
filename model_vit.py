from google.cloud import storage
import os

# ---- CONFIG ----
bucket_name = "bucket-name"
source_blob_name = "best_vit_ham10000.pth"  # file in GCS
destination_file_name = "best_vit_ham10000.pth"  # local path to save

# ---- AUTH ----
# Make sure your environment variable is set:
# export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account.json"
# Or use a JSON file path directly in code (not recommended for public repos)

def download_model():
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    if os.path.exists(destination_file_name):
        print(f"{destination_file_name} already exists locally, skipping download.")
        return

    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {destination_file_name} from bucket {bucket_name}.")

if __name__ == "__main__":
    download_model()
