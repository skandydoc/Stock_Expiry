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
        
        # Common date patterns with year handling
        patterns = [
            # MON-YY
            (r'([A-Z]{3})\s*(\d{2})', lambda m: (m.group(1), '20' + m.group(2))),
            # MON-YYYY
            (r'([A-Z]{3})\s*(\d{4})', lambda m: (m.group(1), m.group(2))),
            # MM-YY
            (r'(\d{2})\s*(\d{2})', lambda m: (datetime.strptime(m.group(1), '%m').strftime('%b'), '20' + m.group(2))),
            # MM-YYYY
            (r'(\d{2})\s*(\d{4})', lambda m: (datetime.strptime(m.group(1), '%m').strftime('%b'), m.group(2)))
        ]
        
        for pattern, handler in patterns:
            match = re.match(pattern, date_str)
            if match:
                month, year = handler(match)
                # Validate year is reasonable (between 2020 and 2040)
                year_int = int(year)
                if not (2020 <= year_int <= 2040):
                    raise ValueError(f"Year {year_int} is out of reasonable range")
                    
                date_str = f"{month} {year}"
                date_obj = datetime.strptime(date_str, '%b %Y')
                # Set to last day of month
                next_month = date_obj.replace(day=28) + pd.DateOffset(months=1)
                last_day = (next_month - pd.DateOffset(days=1)).day
                date_obj = date_obj.replace(day=last_day)
                return date_obj.strftime('%Y-%m-%d')
                
        raise ValueError(f"Unrecognized date format: {date_str}")
    except Exception as e:
        st.warning(f"Could not parse date '{date_str}': {str(e)}")
        return None

@sleep_and_retry
@limits(calls=14, period=60)
def process_image(image):
    """Process image with Gemini model and return structured data"""
    try:
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

        prompt = """
        Analyze this pharmacy inventory image and extract information about all visible medicine packages.
        Focus on finding expiry dates (not manufacturing dates) and accurate locations of brand names.

        For each unique medicine brand visible in the image:
        1. Identify the brand name and its EXACT location in pixels (look for the brand name text only)
        2. Count the number of strips/packages
        3. Note the quantity per strip/package (e.g., 10 tablets, 15 tablets, etc.)
        4. Find the expiry date (specifically look for "EXP", "Expiry", or date after MFG/Mfg date)

        Return the information in this JSON format:
        {
            "items": [
                {
                    "brand_name": "Name of the medicine",
                    "total_packages": number of strips or bottles,
                    "quantity_per_package": number of tablets per strip or volume for syrup,
                    "expiry_date": "found expiry date string",
                    "bounding_box": [x1, y1, x2, y2]
                }
            ]
        }

        Important:
        - Look for EXPIRY dates only, not manufacturing dates
        - For bounding boxes, only mark the exact brand name text location
        - Ensure coordinates are pixel-accurate for the brand name text
        - Count all strips of the same medicine together
        - Report dates exactly as seen on package
        - Double-check all coordinates before returning
        """
        
        response = model.generate_content([prompt, {'mime_type': 'image/png', 'data': img_base64}])
        
        if not response.text:
            st.warning("Response blocked or empty due to safety concerns")
            return None
            
        json_str = response.text.strip()
        if json_str.startswith('```json'):
            json_str = json_str[7:]
        if json_str.endswith('```'):
            json_str = json_str[:-3]
        
        data = json.loads(json_str.strip())
        
        # Validate bounding boxes
        for item in data['items']:
            if 'bounding_box' in item:
                x1, y1, x2, y2 = item['bounding_box']
                if not (0 <= x1 < x2 and 0 <= y1 < y2) or (x2 - x1) > image.width or (y2 - y1) > image.height:
                    item['bounding_box'] = None
                    st.warning(f"Invalid bounding box for {item['brand_name']}")
        
        return data
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def draw_bounding_boxes(image, boxes_data):
    """Draw bounding boxes on the image"""
    img_array = np.array(image)
    img_with_boxes = img_array.copy()
    
    for item in boxes_data['items']:
        if 'bounding_box' in item and item['bounding_box']:
            try:
                x1, y1, x2, y2 = item['bounding_box']
                # Draw rectangle around brand name
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Add brand name label above the box
                label = f"{item['brand_name']}"
                cv2.putText(img_with_boxes, label, (x1, max(y1-10, 20)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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