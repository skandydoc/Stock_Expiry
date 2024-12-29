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

def parse_expiry_date(date_str):
    """Parse various expiry date formats and return YYYY-MM-DD"""
    try:
        # Remove any separators and convert to uppercase
        date_str = re.sub(r'[/\-\.,\s]', ' ', date_str.upper().strip())
        
        # Common date patterns
        patterns = [
            # MON-YY
            (r'([A-Z]{3})\s*(\d{2})', '%b %y'),
            # MON-YYYY
            (r'([A-Z]{3})\s*(\d{4})', '%b %Y'),
            # MM-YY
            (r'(\d{2})\s*(\d{2})', '%m %y'),
            # MM-YYYY
            (r'(\d{2})\s*(\d{4})', '%m %Y')
        ]
        
        for pattern, date_format in patterns:
            match = re.match(pattern, date_str)
            if match:
                # Join the matched groups with space
                date_str = ' '.join(match.groups())
                # Parse the date
                date_obj = datetime.strptime(date_str, date_format)
                # Set day to last day of month for better expiry tracking
                next_month = date_obj.replace(day=28) + pd.DateOffset(months=1)
                last_day = (next_month - pd.DateOffset(days=1)).day
                date_obj = date_obj.replace(day=last_day)
                return date_obj.strftime('%Y-%m-%d')
                
        raise ValueError(f"Unrecognized date format: {date_str}")
    except Exception as e:
        st.warning(f"Error parsing date '{date_str}': {str(e)}. Using end of current month.")
        current_date = datetime.now()
        return current_date.replace(day=1).strftime('%Y-%m-%d')

def parse_quantity(qty_str):
    """Parse quantity string to integer"""
    try:
        # Extract numbers from string
        numbers = re.findall(r'\d+', str(qty_str))
        if numbers:
            return int(numbers[0])
        return 0
    except:
        return 0

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
        For each unique medicine brand visible in the image:
        1. Identify the brand name
        2. Find its location in the image
        3. Count the number of strips/packages
        4. Note the quantity per strip/package
        5. Find the expiry date

        Return the information in this JSON format:
        {
            "items": [
                {
                    "brand_name": "Name of the medicine",
                    "package_type": "strip/syrup/other",
                    "total_packages": number of strips or bottles,
                    "quantity_per_package": number of tablets per strip or volume for syrup,
                    "expiry_date": "found date string",
                    "bounding_box": [x1, y1, x2, y2]
                }
            ]
        }

        Important:
        - Look for ALL unique medicines in the image
        - For strips that are cut, estimate remaining quantity
        - Include partial strips in total_packages count
        - Report the expiry date exactly as seen, don't modify format
        - Bounding box should frame the brand name text
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
        
        return json.loads(json_str.strip())
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def draw_bounding_boxes(image, boxes_data):
    """Draw bounding boxes on the image"""
    img_array = np.array(image)
    img_with_boxes = img_array.copy()
    
    for item in boxes_data['items']:
        if 'bounding_box' in item:
            x1, y1, x2, y2 = item['bounding_box']
            # Draw rectangle around brand name
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add brand name label
            cv2.putText(img_with_boxes, item['brand_name'], (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return Image.fromarray(img_with_boxes)

def create_monthly_summary(data):
    """Create monthly summary of expiring drugs"""
    if not data:
        return pd.DataFrame()
    
    records = []
    for item in data['items']:
        # Calculate total quantity
        total_qty = item.get('total_packages', 0) * item.get('quantity_per_package', 0)
        
        records.append({
            'Brand Name': item['brand_name'],
            'Package Type': item.get('package_type', 'N/A'),
            'Total Packages': item.get('total_packages', 0),
            'Qty per Package': item.get('quantity_per_package', 0),
            'Total Quantity': total_qty,
            'Expiry Date': parse_expiry_date(item.get('expiry_date', ''))
        })
    
    if not records:
        return pd.DataFrame()
        
    df = pd.DataFrame(records)
    df['Expiry Date'] = pd.to_datetime(df['Expiry Date'])
    df['Month-Year'] = df['Expiry Date'].dt.strftime('%Y-%m')
    
    # Group by Month-Year and Brand Name
    summary = df.groupby(['Month-Year', 'Brand Name']).agg({
        'Package Type': 'first',
        'Total Packages': 'sum',
        'Total Quantity': 'sum'
    }).reset_index()
    
    return summary.sort_values('Month-Year')

# Streamlit UI
st.title("Pharmacy Inventory Scanner")
st.write("Upload images of pharmacy inventory to extract and analyze drug information")

uploaded_files = st.file_uploader("Choose image files", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    all_data = []
    
    for uploaded_file in uploaded_files:
        st.write("---")
        st.subheader(f"Processing: {uploaded_file.name}")
        
        # Create two columns for input and output
        col1, col2 = st.columns(2)
        
        # Process image
        image = Image.open(uploaded_file)
        processed_data = process_image(image)
        
        if processed_data:
            # Display original image with bounding boxes
            with col1:
                st.write("Input Image with Detections")
                annotated_image = draw_bounding_boxes(image, processed_data)
                st.image(annotated_image)
            
            # Display structured data as editable table
            with col2:
                st.write("Extracted Information")
                df = pd.DataFrame([{
                    'Brand Name': item['brand_name'],
                    'Package Type': item.get('package_type', 'N/A'),
                    'Total Packages': item.get('total_packages', 0),
                    'Qty per Package': item.get('quantity_per_package', 0),
                    'Total Quantity': item.get('total_packages', 0) * item.get('quantity_per_package', 0),
                    'Expiry Date': item.get('expiry_date', '')
                } for item in processed_data['items']])
                
                edited_df = st.data_editor(df)
                
                # Update processed data with edited values
                for i, row in edited_df.iterrows():
                    if i < len(processed_data['items']):
                        processed_data['items'][i].update({
                            'brand_name': row['Brand Name'],
                            'package_type': row['Package Type'],
                            'total_packages': row['Total Packages'],
                            'quantity_per_package': row['Qty per Package'],
                            'expiry_date': row['Expiry Date']
                        })
            
            all_data.append(processed_data)
    
    if all_data:
        st.write("---")
        st.subheader("Monthly Expiry Summary")
        
        # Combine all data for summary
        combined_data = {'items': [item for data in all_data for item in data['items']]}
        summary_df = create_monthly_summary(combined_data)
        
        if not summary_df.empty:
            st.dataframe(summary_df, use_container_width=True)
        else:
            st.write("No data available for summary")
else:
    st.info("Please upload some images to begin") 