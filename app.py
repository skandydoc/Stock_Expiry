import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os
from dotenv import load_dotenv
import io
import json
from ratelimit import limits, sleep_and_retry
import base64
from datetime import datetime
import re

# Load environment variables
load_dotenv()

# Configure Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    st.error("Please set your GOOGLE_API_KEY in .env file")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def format_month_year(date_str):
    """Convert date string to MON YYYY format"""
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m')
        return date_obj.strftime('%b %Y').upper()
    except:
        return date_str

def parse_expiry_date(date_str):
    """Parse various expiry date formats and return YYYY-MM-DD"""
    if not date_str or date_str.lower() in ['none', 'null', '']:
        return None
        
    try:
        # Remove any separators and convert to uppercase
        date_str = re.sub(r'[/\-\.,\s]+', ' ', date_str.upper().strip())
        
        # First try to extract date after EXP/EXPIRY
        exp_patterns = [
            r'EXP(?:IRY)?[:\s]+([A-Z0-9\s]+)',
            r'EXPIRY\s*DATE\s*:?\s*([A-Z0-9\s]+)',
        ]
        
        for pattern in exp_patterns:
            match = re.search(pattern, date_str)
            if match:
                date_str = match.group(1).strip()
                break
        
        # Common date patterns with year handling
        patterns = [
            # MM YYYY (direct)
            (r'^(\d{1,2})\s+(\d{4})$', lambda m: (int(m.group(1)), int(m.group(2)))),
            # MON-YY
            (r'([A-Z]{3})\s*(\d{2})\b', lambda m: (datetime.strptime(m.group(1), '%b').month, 2000 + int(m.group(2)))),
            # MON-YYYY
            (r'([A-Z]{3})\s*(\d{4})\b', lambda m: (datetime.strptime(m.group(1), '%b').month, int(m.group(2)))),
            # MM-YY
            (r'\b(\d{1,2})\s*(\d{2})\b', lambda m: (int(m.group(1)), 2000 + int(m.group(2)))),
            # MM-YYYY
            (r'\b(\d{1,2})\s*(\d{4})\b', lambda m: (int(m.group(1)), int(m.group(2))))
        ]
        
        for pattern, handler in patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    month, year = handler(match)
                    # Validate month and year
                    if not (1 <= month <= 12):
                        continue
                    if not (2020 <= year <= 2040):
                        continue
                        
                    # Create date object
                    date_obj = datetime(year, month, 1)
                    # Set to last day of month
                    next_month = date_obj.replace(day=28) + pd.DateOffset(months=1)
                    last_day = (next_month - pd.DateOffset(days=1)).day
                    date_obj = date_obj.replace(day=last_day)
                    return date_obj.strftime('%Y-%m-%d')
                except:
                    continue
                
        raise ValueError(f"Could not parse date format: {date_str}")
    except Exception as e:
        st.warning(f"Could not parse date '{date_str}': {str(e)}")
        return None

def normalize_coordinates(box, image_width, image_height):
    """Convert pixel coordinates to normalized coordinates (0-1000 range)"""
    x1, y1, x2, y2 = box
    return [
        int(y1 * 1000 / image_height),  # ymin
        int(x1 * 1000 / image_width),   # xmin
        int(y2 * 1000 / image_height),  # ymax
        int(x2 * 1000 / image_width)    # xmax
    ]

def denormalize_coordinates(box, image_width, image_height):
    """Convert normalized coordinates (0-1000 range) to pixel coordinates"""
    ymin, xmin, ymax, xmax = box
    return [
        int(xmin * image_width / 1000),   # x1
        int(ymin * image_height / 1000),  # y1
        int(xmax * image_width / 1000),   # x2
        int(ymax * image_height / 1000)   # y2
    ]

def validate_coordinates(box, image_width, image_height, min_size=10):
    """Validate and normalize bounding box coordinates"""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
        
    try:
        # Convert normalized coordinates to pixel coordinates
        x1, y1, x2, y2 = denormalize_coordinates(box, image_width, image_height)
        
        # Ensure correct ordering
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Check minimum size
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            return None
            
        # Check maximum size (shouldn't be more than half the image)
        if (x2 - x1) > image_width/2 or (y2 - y1) > image_height/2:
            return None
            
        return [x1, y1, x2, y2]
    except:
        return None

@sleep_and_retry
@limits(calls=14, period=60)
def process_image(image):
    """Process image with Gemini model and return structured data"""
    try:
        # Get original image dimensions
        original_width, original_height = image.size
        
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

        prompt = """
        Analyze this pharmacy inventory image ({width}x{height} pixels) and extract information about all visible medicine packages.
        Pay special attention to distinguishing between brand names and generic names.

        For each unique medicine visible in the image:
        1. Find both:
           - Brand/Trade name (e.g., Allegra-M, Pantosec)
           - Generic name (chemical name of the drug)
        2. Count all strips/packages of this medicine
        3. Note the quantity per strip (usually 10, 15, 20, or 30 tablets)
        4. Find dates, distinguishing between:
           - Manufacturing Date (MFG/Mfg Date)
           - Expiry Date (EXP/Expiry Date)
        5. Find the exact location of the brand name text in the image

        Return a valid JSON object in this exact format:
        {{
            "items": [
                {{
                    "brand_name": "Brand/Trade name of the medicine",
                    "generic_name": "Generic/Chemical name of the drug",
                    "total_packages": number of strips (count partial strips as 1),
                    "quantity_per_package": number of tablets per strip (10/15/20/30),
                    "mfg_date": "manufacturing date exactly as shown",
                    "expiry_date": "expiry date exactly as shown",
                    "bounding_box": [ymin, xmin, ymax, xmax]
                }}
            ]
        }}

        Important:
        - Brand name should be the trade name of the drug (NOT the manufacturer name like Cipla/Leeford)
        - Generic name should be the chemical/scientific name of the drug
        - Return coordinates in normalized format (0-1000 range)
        - Coordinates should be in [ymin, xmin, ymax, xmax] format
        - Brand name bounding box should tightly frame only the brand name text
        - All numbers must be integers
        - Report dates exactly as shown on package
        """.format(width=original_width, height=original_height)
        
        response = model.generate_content([prompt, {'mime_type': 'image/png', 'data': img_base64}])
        
        if not response.text:
            st.warning("Response blocked or empty due to safety concerns")
            return None
            
        # Clean and parse JSON response
        json_str = response.text.strip()
        json_str = re.sub(r'^```json\s*', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'\s*```$', '', json_str, flags=re.MULTILINE)
        json_str = json_str.strip()
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON response: {str(e)}")
            st.code(json_str, language='json')
            return None
            
        if not isinstance(data, dict) or 'items' not in data or not isinstance(data['items'], list):
            st.error("Invalid response format: missing 'items' array")
            return None
        
        # Validate and normalize data
        for item in data['items']:
            if not isinstance(item, dict):
                continue
                
            # Ensure required fields exist
            item['brand_name'] = str(item.get('brand_name', '')).strip()
            item['generic_name'] = str(item.get('generic_name', '')).strip()
            item['total_packages'] = int(item.get('total_packages', 0))
            item['quantity_per_package'] = int(item.get('quantity_per_package', 0))
            
            # Validate bounding box
            if 'bounding_box' in item:
                box = item['bounding_box']
                if not isinstance(box, list) or len(box) != 4:
                    item['bounding_box'] = None
                    continue
                    
                try:
                    # Ensure coordinates are within 0-1000 range
                    ymin, xmin, ymax, xmax = map(float, box)
                    if not all(0 <= coord <= 1000 for coord in [ymin, xmin, ymax, xmax]):
                        item['bounding_box'] = None
                        continue
                        
                    if ymin >= ymax or xmin >= xmax:
                        item['bounding_box'] = None
                        continue
                        
                    item['bounding_box'] = [int(ymin), int(xmin), int(ymax), int(xmax)]
                except:
                    item['bounding_box'] = None
            
            # Validate quantity
            try:
                qty = int(item['quantity_per_package'])
                if qty not in [10, 15, 20, 30]:
                    st.warning(f"Unusual quantity per package for {item['brand_name']}: {qty}")
            except:
                item['quantity_per_package'] = 0
        
        return data
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def draw_bounding_boxes(image, boxes_data):
    """Draw bounding boxes on the image"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    img_with_boxes = img_array.copy()
    
    # Get image dimensions
    height, width = img_with_boxes.shape[:2]
    
    # Define colors for different elements
    colors = {
        'brand': (0, 255, 0),  # Green for brand names
        'text': (255, 255, 255)  # White for text background
    }
    
    for item in boxes_data['items']:
        if 'bounding_box' in item and item['bounding_box']:
            try:
                # Convert normalized coordinates to pixel coordinates
                x1, y1, x2, y2 = denormalize_coordinates(item['bounding_box'], width, height)
                
                # Validate coordinates
                if not (0 <= x1 < width and 0 <= y1 < height and 0 <= x2 < width and 0 <= y2 < height):
                    st.warning(f"Invalid coordinates for {item['brand_name']}: outside image bounds")
                    continue
                
                if x2 <= x1 or y2 <= y1:
                    st.warning(f"Invalid coordinates for {item['brand_name']}: negative or zero area")
                    continue
                
                # Draw rectangle with slightly thicker lines for visibility
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), colors['brand'], 2)
                
                # Prepare label text
                label = item['brand_name']
                
                # Calculate text size and position
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Position text above the box if possible, below if not enough space
                if y1 > text_height + 10:
                    text_y = y1 - 5
                else:
                    text_y = y2 + text_height + 5
                
                # Create background for text
                padding = 2
                bg_pts = np.array([
                    [x1, text_y - text_height - padding],
                    [x1 + text_width + 2*padding, text_y - text_height - padding],
                    [x1 + text_width + 2*padding, text_y + padding],
                    [x1, text_y + padding]
                ], np.int32)
                
                # Draw semi-transparent background
                overlay = img_with_boxes.copy()
                cv2.fillPoly(overlay, [bg_pts], colors['text'])
                cv2.addWeighted(overlay, 0.7, img_with_boxes, 0.3, 0, img_with_boxes)
                
                # Draw text
                cv2.putText(img_with_boxes, label, (x1 + padding, text_y),
                           font, font_scale, colors['brand'], thickness)
                
            except Exception as e:
                st.warning(f"Error drawing box for {item['brand_name']}: {str(e)}")
    
    return Image.fromarray(img_with_boxes)

def create_monthly_summary(data):
    """Create monthly summary of expiring drugs"""
    if not data or not data.get('items'):
        return pd.DataFrame()
    
    records = []
    for item in data['items']:
        expiry_date = parse_expiry_date(item.get('expiry_date', ''))
        if expiry_date:  # Only include items with valid expiry dates
            total_qty = item.get('total_packages', 0) * item.get('quantity_per_package', 0)
            records.append({
                'Brand Name': item['brand_name'],
                'Generic Name': item.get('generic_name', ''),
                'Total Packages': item.get('total_packages', 0),
                'Total Quantity': total_qty,
                'Expiry Date': expiry_date
            })
    
    if not records:
        return pd.DataFrame()
        
    df = pd.DataFrame(records)
    df['Expiry Date'] = pd.to_datetime(df['Expiry Date'])
    df['Month-Year'] = df['Expiry Date'].dt.strftime('%Y-%m')
    
    # Group by Month-Year and Brand Name
    summary = df.groupby(['Month-Year', 'Brand Name', 'Generic Name']).agg({
        'Total Packages': 'sum',
        'Total Quantity': 'sum'
    }).reset_index()
    
    return summary.sort_values('Month-Year')

# Streamlit UI
st.title("Pharmacy Inventory Scanner")
st.write("Upload images of pharmacy inventory to extract and analyze drug information")

# Add sample image button
if st.button("Try with Sample Image"):
    try:
        with open("image.png", "rb") as f:
            uploaded_files = [io.BytesIO(f.read())]
            uploaded_files[0].name = "image.png"
    except Exception as e:
        st.error(f"Error loading sample image: {str(e)}")
        uploaded_files = []
else:
    uploaded_files = st.file_uploader("Choose image files", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    all_data = []
    
    for uploaded_file in uploaded_files:
        try:
            # Process image
            image = Image.open(uploaded_file)
            processed_data = process_image(image)
            
            if processed_data:
                st.write("---")
                st.subheader(f"Processing: {uploaded_file.name}")
                
                # Display original image with bounding boxes
                st.write("Input Image with Detections")
                annotated_image = draw_bounding_boxes(image, processed_data)
                st.image(annotated_image)
                
                # Display structured data as editable table
                st.write("Extracted Information")
                df = pd.DataFrame([{
                    'Brand Name': item['brand_name'],
                    'Generic Name': item.get('generic_name', ''),
                    'Total Packages': item.get('total_packages', 0),
                    'Qty per Package': item.get('quantity_per_package', 0),
                    'Total Quantity': item.get('total_packages', 0) * item.get('quantity_per_package', 0),
                    'Expiry Date': item.get('expiry_date', '')
                } for item in processed_data['items']])
                
                # Add index starting from 1
                df.index = range(1, len(df) + 1)
                
                # Display full table without horizontal scroll
                edited_df = st.data_editor(
                    df,
                    use_container_width=True,
                    num_rows="fixed",
                    hide_index=False,
                    column_config={
                        'Brand Name': st.column_config.TextColumn('Brand Name', width='medium'),
                        'Generic Name': st.column_config.TextColumn('Generic Name', width='medium'),
                        'Total Packages': st.column_config.NumberColumn('Total Packages', width='small'),
                        'Qty per Package': st.column_config.NumberColumn('Qty per Package', width='small'),
                        'Total Quantity': st.column_config.NumberColumn('Total Quantity', width='small'),
                        'Expiry Date': st.column_config.TextColumn('Expiry Date', width='medium')
                    }
                )
                
                # Update processed data with edited values
                for i, row in edited_df.iterrows():
                    idx = i - 1  # Convert 1-based index back to 0-based
                    if idx < len(processed_data['items']):
                        processed_data['items'][idx].update({
                            'brand_name': row['Brand Name'],
                            'generic_name': row['Generic Name'],
                            'total_packages': row['Total Packages'],
                            'quantity_per_package': row['Qty per Package'],
                            'expiry_date': row['Expiry Date']
                        })
                
                all_data.append(processed_data)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            continue
    
    if all_data:
        st.write("---")
        st.subheader("Monthwise Expiry List Summary")
        
        try:
            # Combine all data for summary
            combined_data = {'items': [item for data in all_data for item in data['items']]}
            summary_df = create_monthly_summary(combined_data)
            
            if not summary_df.empty:
                # Group by Month-Year and display separate tables
                for month_year in sorted(summary_df['Month-Year'].unique()):
                    month_data = summary_df[summary_df['Month-Year'] == month_year].copy()
                    month_data.index = range(1, len(month_data) + 1)  # Start index from 1
                    
                    formatted_date = format_month_year(month_year)
                    st.write(f"**Expiring in {formatted_date}**")
                    st.dataframe(
                        month_data.drop('Month-Year', axis=1),
                        use_container_width=True,
                        hide_index=False
                    )
            else:
                st.info("No expiry data available for summary")
        except Exception as e:
            st.error(f"Error creating summary: {str(e)}")
else:
    st.info("Please upload some images to begin") 