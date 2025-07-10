import boto3
import os
import time
from src import config
from dotenv import load_dotenv

# Load AWS credentials
load_dotenv()
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")

upload_dir_path=config.PATHS.get("upload_dir_path","")
os.makedirs(upload_dir_path, exist_ok=True)

def save_file_to_local(pdf_file,session_id):
    safe_filename = f"{pdf_file.name.replace('.pdf','')}_{session_id}.pdf"
    pdf_path = os.path.join(upload_dir_path, safe_filename)
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.getvalue())
    return pdf_path

# Upload PDF to S3
def upload_to_s3(pdf_path):
    # File & S3 Info
    PDF_FILE = pdf_path
    S3_BUCKET = 'spsoft-aiml-workspace'  # 📝 Fill in your bucket name
    S3_OBJECT_NAME = 'uploaded_1040.pdf'
    
    # Boto3 Clients
    s3 = boto3.client('s3', region_name=aws_region,
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key)

    textract = boto3.client('textract', region_name=aws_region,
                            aws_access_key_id=aws_access_key,
                            aws_secret_access_key=aws_secret_key)

    s3.upload_file(PDF_FILE, S3_BUCKET, S3_OBJECT_NAME)
    print(f"Uploaded {PDF_FILE} to S3 bucket {S3_BUCKET}")

    # Start Textract analysis
    response = textract.start_document_analysis(
        DocumentLocation={'S3Object': {'Bucket': S3_BUCKET, 'Name': S3_OBJECT_NAME}},
        FeatureTypes=["FORMS"]
    )
    job_id = response['JobId']
    print(f"Started Textract job: {job_id}")
    return job_id,textract

# Wait for job to complete
def wait_for_textract(job_id,textract):
    while True:
        result = textract.get_document_analysis(JobId=job_id)
        status = result['JobStatus']
        if status == 'SUCCEEDED':
            return
        elif status == 'FAILED':
            raise Exception("Textract job failed.")
        print("Waiting for Textract job to complete...")
        time.sleep(5)

# Get all pages
def get_all_results(job_id,textract):
    pages = []
    next_token = None
    while True:
        if next_token:
            response = textract.get_document_analysis(JobId=job_id, NextToken=next_token)
        else:
            response = textract.get_document_analysis(JobId=job_id)
        pages.append(response)
        next_token = response.get('NextToken')
        if not next_token:
            break
    return pages

# Extract LINE text blocks
def extract_raw_text(pages):
    lines = []
    for page in pages:
        for block in page['Blocks']:
            if block['BlockType'] == 'LINE':
                lines.append(block['Text'])
    return "\n".join(lines)

# Execute
def extract_data(pdf_file,session_id):
    pdf_path=save_file_to_local(pdf_file,session_id)
    job_id,textract=upload_to_s3(pdf_path)
    wait_for_textract(job_id,textract)
    pages = get_all_results(job_id,textract)
    raw_text = extract_raw_text(pages)
    text_file_name=f'{session_id}_text.txt'
    OUTPUT_TEXT_FILE=config.PATHS.get("raw_text_path","")
    text_path=os.path.join(OUTPUT_TEXT_FILE, text_file_name)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(raw_text)
    print(f"✅ Raw text extracted and saved to: {text_path}")
    return text_path
