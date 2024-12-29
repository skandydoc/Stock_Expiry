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
        date_str = re.sub(r'[/\-\.,\s]', ' ', date_str.upper().strip())
        
        # Extract date if it's after EXP or EXPIRY
        exp_match = re.search(r'EXP(?:IRY)?\s*:?\s*([A-Z0-9\s]+)', date_str)
        if exp_match:
            date_str = exp_match.group(1).strip()
        
        # Common date patterns with year handling
        patterns = [
            # MON-YY
            (r'([A-Z]{3})\s*(\d{2})\b', lambda m: (m.group(1), '20' + m.group(2))),
            # MON-YYYY
            (r'([A-Z]{3})\s*(\d{4})\b', lambda m: (m.group(1), m.group(2))),
            # MM-YY
            (r'\b(\d{2})\s*(\d{2})\b', lambda m: (datetime.strptime(m.group(1), '%m').strftime('%b'), '20' + m.group(2))),
            # MM-YYYY
            (r'\b(\d{2})\s*(\d{4})\b', lambda m: (datetime.strptime(m.group(1), '%m').strftime('%b'), m.group(2)))
        ]
        
        for pattern, handler in patterns:
            match = re.search(pattern, date_str)
            if match:
                month, year = handler(match)
                # Validate year is reasonable (between 2020 and 2040)
                year_int = int(year)
                if not (2020 <= year_int <= 2040):
                    continue  # Try next pattern if year is unreasonable
                    
                try:
                    date_str = f"{month} {year}"
                    date_obj = datetime.strptime(date_str, '%b %Y')
                    # Set to last day of month
                    next_month = date_obj.replace(day=28) + pd.DateOffset(months=1)
                    last_day = (next_month - pd.DateOffset(days=1)).day
                    date_obj = date_obj.replace(day=last_day)
                    return date_obj.strftime('%Y-%m-%d')
                except:
                    continue
                
        raise ValueError(f"Unrecognized date format: {date_str}")
    except Exception as e:
        st.warning(f"Could not parse date '{date_str}': {str(e)}")
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
        Analyze this pharmacy inventory image and extract information about all visible medicine packages.
        The image dimensions are {width}x{height} pixels.

        For each unique medicine brand visible in the image:
        1. Identify the brand name and its EXACT pixel coordinates in the image
        2. Count the number of complete and partial strips/packages
        3. Look for numbers indicating tablets per strip (usually 10, 15, 20, or 30)
        4. Find the expiry date by looking for:
           - Text starting with "EXP" or "Expiry"
           - Date format after "Mfg." or "Manufacturing Date"
           - Usually in format MM/YY, MM/YYYY, MON/YY, or MON/YYYY

        Return a valid JSON object in this exact format:
        {{
            "items": [
                {{
                    "brand_name": "Exact brand name as shown",
                    "total_packages": number of strips (count partial strips as 1),
                    "quantity_per_package": number of tablets per strip (10/15/20/30),
                    "expiry_date": "exact expiry date as shown on package",
                    "bounding_box": [x1, y1, x2, y2]
                }}
            ]
        }}

        Important:
        - Return ONLY the JSON object, no additional text or formatting
        - Ensure all numbers are integers
        - Coordinates must be within image dimensions: 0 ≤ x ≤ {width}, 0 ≤ y ≤ {height}
        - Only mark the exact brand name text location
        - Count all strips of the same medicine together
        - Report the exact date text as seen, don't modify the format
        """.format(width=original_width, height=original_height)
        
        response = model.generate_content([prompt, {'mime_type': 'image/png', 'data': img_base64}])
        
        if not response.text:
            st.warning("Response blocked or empty due to safety concerns")
            return None
            
        # Clean and parse JSON response
        json_str = response.text.strip()
        # Remove any markdown code block syntax
        json_str = re.sub(r'^```json\s*', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'\s*```$', '', json_str, flags=re.MULTILINE)
        json_str = json_str.strip()
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON response: {str(e)}")
            st.code(json_str, language='json')  # Display the problematic JSON
            return None
            
        if not isinstance(data, dict) or 'items' not in data or not isinstance(data['items'], list):
            st.error("Invalid response format: missing 'items' array")
            return None
        
        # Validate and normalize bounding boxes
        for item in data['items']:
            if not isinstance(item, dict):
                continue
                
            # Ensure required fields exist
            item['brand_name'] = str(item.get('brand_name', ''))
            item['total_packages'] = int(item.get('total_packages', 0))
            item['quantity_per_package'] = int(item.get('quantity_per_package', 0))
            item['expiry_date'] = str(item.get('expiry_date', ''))
            
            if 'bounding_box' in item:
                try:
                    box = item['bounding_box']
                    if not isinstance(box, list) or len(box) != 4:
                        raise ValueError("Invalid box format")
                        
                    x1, y1, x2, y2 = map(int, box)
                    # Ensure coordinates are within image bounds
                    x1 = max(0, min(x1, original_width))
                    x2 = max(0, min(x2, original_width))
                    y1 = max(0, min(y1, original_height))
                    y2 = max(0, min(y2, original_height))
                    
                    # Ensure box has reasonable size
                    if (x2 - x1) < 10 or (y2 - y1) < 5 or (x2 - x1) > original_width/2:
                        item['bounding_box'] = None
                        st.warning(f"Invalid box size for {item['brand_name']}")
                    else:
                        item['bounding_box'] = [x1, y1, x2, y2]
                except Exception as e:
                    item['bounding_box'] = None
                    st.warning(f"Invalid coordinates for {item['brand_name']}: {str(e)}")
            
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
    img_array = np.array(image)
    img_with_boxes = img_array.copy()
    
    # Get image dimensions
    height, width = img_with_boxes.shape[:2]
    
    for item in boxes_data['items']:
        if 'bounding_box' in item and item['bounding_box']:
            try:
                x1, y1, x2, y2 = item['bounding_box']
                # Ensure coordinates are within bounds
                x1, x2 = min(max(0, x1), width), min(max(0, x2), width)
                y1, y2 = min(max(0, y1), height), min(max(0, y2), height)
                
                # Draw rectangle around brand name
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add brand name label above the box
                label = f"{item['brand_name']}"
                # Calculate text size and position
                font_scale = 0.6
                thickness = 2
                font = cv2.FONT_HERSHEY_SIMPLEX
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Ensure label is visible
                text_x = x1
                text_y = max(text_height + 5, y1 - 5)
                
                # Draw white background for text
                cv2.rectangle(img_with_boxes, 
                            (text_x, text_y - text_height - 5),
                            (text_x + text_width, text_y + 5),
                            (255, 255, 255), -1)
                            
                cv2.putText(img_with_boxes, label, (text_x, text_y),
                           font, font_scale, (0, 255, 0), thickness)
                           
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
    summary = df.groupby(['Month-Year', 'Brand Name']).agg({
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
                    hide_index=False
                )
                
                # Update processed data with edited values
                for i, row in edited_df.iterrows():
                    idx = i - 1  # Convert 1-based index back to 0-based
                    if idx < len(processed_data['items']):
                        processed_data['items'][idx].update({
                            'brand_name': row['Brand Name'],
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