import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATHS = {
    "upload_dir_path": os.path.join(BASE_DIR, "upload_files"),
    "raw_text_path": os.path.join(BASE_DIR, "extracted_raw_data"),
    "section_save_path": os.path.join(BASE_DIR, "sectioned_data"),
    "recommendation_json_path": os.path.join(BASE_DIR, "recommendations_json_data"),
    "json_data_path": os.path.join(BASE_DIR, "extracted_json_data")
}
  