import boto3
import time
import json
import os
from src import config
from dotenv import load_dotenv
from collections import defaultdict
import re

upload_dir_path=config.PATHS.get("upload_dir_path","")
os.makedirs(upload_dir_path, exist_ok=True)

load_dotenv()
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")

class LLMOptimizedTextractExtractor:
    def __init__(self, bucket_name):
        self.s3 = boto3.client('s3', region_name=aws_region,
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key)
        self.textract = boto3.client('textract', region_name=aws_region,
                            aws_access_key_id=aws_access_key,
                            aws_secret_access_key=aws_secret_key)
        self.bucket_name = bucket_name
        
    def extract_for_llm_processing(self, pdf_path, s3_object_name):
        """
        Extract text optimized for LLM-based field mapping and sectioning
        This method focuses on preserving logical structure over exact visual positioning
        """
        # Upload PDF to S3
        self.s3.upload_file(pdf_path, self.bucket_name, s3_object_name)
        
        # Start Textract Job with FORMS and TABLES features
        response = self.textract.start_document_analysis(
            DocumentLocation={
                'S3Object': {
                    'Bucket': self.bucket_name,
                    'Name': s3_object_name
                }
            },
            FeatureTypes=["FORMS", "TABLES"]
        )
        job_id = response["JobId"]
        print(f"Started Textract job with ID: {job_id}")
        
        # Wait for job completion
        while True:
            result = self.textract.get_document_analysis(JobId=job_id)
            status = result["JobStatus"]
            if status in ["SUCCEEDED", "FAILED"]:
                break
            print("Waiting for Textract job to finish...")
            time.sleep(5)
            
        if status != "SUCCEEDED":
            raise Exception("Textract job failed.")
            
        # Get all pages (handle pagination)
        blocks = []
        next_token = None
        while True:
            kwargs = {"JobId": job_id}
            if next_token:
                kwargs["NextToken"] = next_token
                
            response = self.textract.get_document_analysis(**kwargs)
            blocks.extend(response["Blocks"])
            next_token = response.get("NextToken")
            if not next_token:
                break
                
        return self._process_for_llm_sectioning(blocks)
    
    def _process_for_llm_sectioning(self, blocks):
        """
        Process blocks optimized for LLM sectioning and field mapping
        """
        # Create lookup tables
        block_map = {block['Id']: block for block in blocks}
        
        # Separate by pages first
        pages = defaultdict(list)
        for block in blocks:
            page_num = block.get('Page', 1)
            pages[page_num].append(block)
        
        all_pages_content = []
        
        for page_num in sorted(pages.keys()):
            page_content = self._process_page_for_llm(pages[page_num], block_map)
            all_pages_content.append({
                'page': page_num,
                'content': page_content
            })
        
        # Combine all pages into structured text
        return self._format_for_llm_consumption(all_pages_content)
    
    def _process_page_for_llm(self, page_blocks, block_map):
        """
        Process a single page optimized for LLM understanding
        """
        # Collect different types of content
        form_fields = []
        tables = []
        standalone_text = []
        
        processed_blocks = set()
        
        # Process form fields (key-value pairs)
        for block in page_blocks:
            if block['Id'] in processed_blocks:
                continue
                
            if block['BlockType'] == 'KEY_VALUE_SET' and 'KEY' in block.get('EntityTypes', []):
                key_text = self._get_text_from_block(block, block_map)
                value_text = ""
                
                # Find associated value
                for relationship in block.get('Relationships', []):
                    if relationship['Type'] == 'VALUE':
                        for value_id in relationship['Ids']:
                            value_block = block_map.get(value_id)
                            if value_block:
                                value_text = self._get_text_from_block(value_block, block_map)
                                processed_blocks.add(value_id)
                                break
                
                if key_text or value_text:
                    form_fields.append({
                        'type': 'form_field',
                        'key': key_text.strip(),
                        'value': value_text.strip(),
                        'position': block['Geometry']['BoundingBox']['Top']
                    })
                
                processed_blocks.add(block['Id'])
        
        # Process tables
        for block in page_blocks:
            if block['Id'] in processed_blocks:
                continue
                
            if block['BlockType'] == 'TABLE':
                table_data = self._extract_table_for_llm(block, block_map)
                if table_data:
                    tables.append({
                        'type': 'table',
                        'data': table_data,
                        'position': block['Geometry']['BoundingBox']['Top']
                    })
                
                # Mark all table cells as processed
                for relationship in block.get('Relationships', []):
                    if relationship['Type'] == 'CHILD':
                        for child_id in relationship['Ids']:
                            processed_blocks.add(child_id)
                            # Also mark cell children as processed
                            child_block = block_map.get(child_id)
                            if child_block:
                                for child_rel in child_block.get('Relationships', []):
                                    if child_rel['Type'] == 'CHILD':
                                        processed_blocks.update(child_rel['Ids'])
        
        # Process remaining standalone elements
        standalone_elements = []
        for block in page_blocks:
            if block['Id'] in processed_blocks:
                continue
                
            if block['BlockType'] in ['WORD', 'SELECTION_ELEMENT']:
                bbox = block['Geometry']['BoundingBox']
                
                if block['BlockType'] == 'WORD':
                    text = block['Text']
                else:
                    text = "[X]" if block['SelectionStatus'] == 'SELECTED' else "[ ]"
                
                standalone_elements.append({
                    'text': text,
                    'type': block['BlockType'],
                    'top': bbox['Top'],
                    'left': bbox['Left']
                })
        
        # Group standalone elements into logical lines
        if standalone_elements:
            lines = self._group_standalone_elements(standalone_elements)
            for line in lines:
                standalone_text.append({
                    'type': 'text_line',
                    'content': line['content'],
                    'position': line['position']
                })
        
        # Combine all content types
        all_content = form_fields + tables + standalone_text
        
        # Sort by vertical position
        all_content.sort(key=lambda x: x['position'])
        
        return all_content
    
    def _extract_table_for_llm(self, table_block, block_map):
        """
        Extract table data in a format that's easy for LLMs to understand
        """
        if 'Relationships' not in table_block:
            return None
        
        # Get all cells
        cells = []
        for relationship in table_block['Relationships']:
            if relationship['Type'] == 'CHILD':
                for cell_id in relationship['Ids']:
                    cell_block = block_map.get(cell_id)
                    if cell_block and cell_block['BlockType'] == 'CELL':
                        cell_text = self._get_text_from_block(cell_block, block_map)
                        cells.append({
                            'text': cell_text,
                            'row': cell_block.get('RowIndex', 1),
                            'col': cell_block.get('ColumnIndex', 1)
                        })
        
        if not cells:
            return None
        
        # Organize cells into rows and columns
        max_row = max(cell['row'] for cell in cells)
        max_col = max(cell['col'] for cell in cells)
        
        table_array = [['' for _ in range(max_col)] for _ in range(max_row)]
        
        for cell in cells:
            table_array[cell['row'] - 1][cell['col'] - 1] = cell['text']
        
        # Convert to text format that's LLM-friendly
        table_text = []
        for row in table_array:
            # Join cells with | separator and clean up
            row_text = ' | '.join(cell.strip() for cell in row)
            table_text.append(row_text)
        
        return '\n'.join(table_text)
    
    def _get_text_from_block(self, block, block_map):
        """
        Extract all text content from a block and its children
        """
        text_parts = []
        
        if 'Relationships' in block:
            for relationship in block['Relationships']:
                if relationship['Type'] == 'CHILD':
                    for child_id in relationship['Ids']:
                        child_block = block_map.get(child_id)
                        if child_block:
                            if child_block['BlockType'] == 'WORD':
                                text_parts.append(child_block['Text'])
                            elif child_block['BlockType'] == 'SELECTION_ELEMENT':
                                checkbox = "[X]" if child_block['SelectionStatus'] == 'SELECTED' else "[ ]"
                                text_parts.append(checkbox)
        
        return ' '.join(text_parts)
    
    def _group_standalone_elements(self, elements, line_tolerance=0.01):
        """
        Group standalone elements into logical lines for better LLM understanding
        """
        if not elements:
            return []
        
        # Sort by position
        elements.sort(key=lambda x: (x['top'], x['left']))
        
        lines = []
        current_line = [elements[0]]
        current_top = elements[0]['top']
        
        for element in elements[1:]:
            if abs(element['top'] - current_top) <= line_tolerance:
                current_line.append(element)
            else:
                # Process current line
                current_line.sort(key=lambda x: x['left'])
                line_content = ' '.join(elem['text'] for elem in current_line)
                lines.append({
                    'content': line_content,
                    'position': current_top
                })
                
                # Start new line
                current_line = [element]
                current_top = element['top']
        
        # Don't forget the last line
        if current_line:
            current_line.sort(key=lambda x: x['left'])
            line_content = ' '.join(elem['text'] for elem in current_line)
            lines.append({
                'content': line_content,
                'position': current_top
            })
        
        return lines
    
    def _format_for_llm_consumption(self, pages_content):
        """
        Format the extracted content for optimal LLM processing
        """
        formatted_output = []
        
        for page_info in pages_content:
            page_num = page_info['page']
            content = page_info['content']
            
            formatted_output.append(f"=== PAGE {page_num} ===")
            
            for item in content:
                if item['type'] == 'form_field':
                    if item['key'] and item['value']:
                        formatted_output.append(f"{item['key']}: {item['value']}")
                    elif item['key']:
                        formatted_output.append(f"{item['key']}")
                    elif item['value']:
                        formatted_output.append(f"{item['value']}")
                
                elif item['type'] == 'table':
                    formatted_output.append("TABLE:")
                    formatted_output.append(item['data'])
                    formatted_output.append("")  # Add spacing after tables
                
                elif item['type'] == 'text_line':
                    formatted_output.append(item['content'])
            
            formatted_output.append("")  # Add spacing between pages
        
        return '\n'.join(formatted_output)
    
    def detect_all_schedules(self, text):
        """
        Dynamically detect all schedules present in the document
        """
        # Common schedule patterns
        main_schedules = {
            'A': r'SCHEDULE A\s*\([^)]*\)',
            'B': r'SCHEDULE B\s*\([^)]*\)',
            'C': r'SCHEDULE C\s*\([^)]*\)',
            'D': r'SCHEDULE D\s*\([^)]*\)',
            'E': r'SCHEDULE E\s*\([^)]*\)',
            'F': r'SCHEDULE F\s*\([^)]*\)',
            'H': r'SCHEDULE H\s*\([^)]*\)',
            'J': r'SCHEDULE J\s*\([^)]*\)',
            'K-1': r'SCHEDULE K-1\s*\([^)]*\)',
            'R': r'SCHEDULE R\s*\([^)]*\)',
            'SE': r'SCHEDULE SE\s*\([^)]*\)',
            '1': r'SCHEDULE 1\s*\([^)]*\)',
            '2': r'SCHEDULE 2\s*\([^)]*\)',
            '3': r'SCHEDULE 3\s*\([^)]*\)',
            '8812': r'SCHEDULE 8812\s*\([^)]*\)',
            'LEP': r'SCHEDULE LEP\s*\([^)]*\)'
        }
        
        # Additional forms that might be attached
        additional_forms = {
            '2441': r'FORM 2441\s*\([^)]*\)',
            '4562': r'FORM 4562\s*\([^)]*\)',
            '4684': r'FORM 4684\s*\([^)]*\)',
            '4797': r'FORM 4797\s*\([^)]*\)',
            '4868': r'FORM 4868\s*\([^)]*\)',
            '5329': r'FORM 5329\s*\([^)]*\)',
            '6251': r'FORM 6251\s*\([^)]*\)',
            '8283': r'FORM 8283\s*\([^)]*\)',
            '8582': r'FORM 8582\s*\([^)]*\)',
            '8606': r'FORM 8606\s*\([^)]*\)',
            '8829': r'FORM 8829\s*\([^)]*\)',
            '8949': r'FORM 8949\s*\([^)]*\)'
        }
        
        detected_schedules = {}
        detected_forms = {}
        
        # Check for main schedules
        for schedule, pattern in main_schedules.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected_schedules[schedule] = {
                    'pattern': pattern,
                    'matches': matches,
                    'count': len(matches)
                }
        
        # Check for additional forms
        for form, pattern in additional_forms.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected_forms[form] = {
                    'pattern': pattern,
                    'matches': matches,
                    'count': len(matches)
                }
        
        return detected_schedules, detected_forms

def save_file_to_local(pdf_file,session_id):
    safe_filename = f"{pdf_file.name.replace('.pdf','')}_{session_id}.pdf"
    pdf_path = os.path.join(upload_dir_path, safe_filename)
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.getvalue())
    return pdf_path


def extract_data(pdf_file,session_id):
    # Test the extraction
    bucket_name = 'spsoft-aiml-workspace'
    s3_object_name = 'uploaded_1040.pdf'
    
    extractor = LLMOptimizedTextractExtractor(bucket_name)
    pdf_path=save_file_to_local(pdf_file,session_id)
    print("file saved to local")
    # Method 1: Basic LLM-optimized extraction
    text_for_llm = extractor.extract_for_llm_processing(pdf_path, s3_object_name)
    # print("=== LLM-OPTIMIZED EXTRACTION ===")
    text_file_name=f'{session_id}_text.txt'
    OUTPUT_TEXT_FILE=config.PATHS.get("raw_text_path","")
    text_path=os.path.join(OUTPUT_TEXT_FILE, text_file_name)
    # Save result
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text_for_llm)
    return text_path

