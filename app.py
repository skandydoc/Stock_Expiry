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

# Disable usage stats
st.set_option('browser.gatherUsageStats', False)

# Load environment variables
load_dotenv()

# Configure Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    st.error("Please set your GOOGLE_API_KEY in .env file")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

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
        Analyze this pharmacy inventory image and extract the following information in JSON format:
        - Drug generic name
        - Brand name
        - Expiry date (in YYYY-MM-DD format)
        - Quantity
        Also provide bounding box coordinates for each text element in format [x1, y1, x2, y2].
        Return the response as a valid JSON string with the following structure:
        {
            "items": [
                {
                    "generic_name": "...",
                    "brand_name": "...",
                    "expiry_date": "YYYY-MM-DD",
                    "quantity": "...",
                    "bounding_boxes": {
                        "generic_name": [x1, y1, x2, y2],
                        "brand_name": [x1, y1, x2, y2],
                        "expiry_date": [x1, y1, x2, y2],
                        "quantity": [x1, y1, x2, y2]
                    }
                }
            ]
        }
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
    
    colors = {
        'generic_name': (255, 0, 0),    # Red
        'brand_name': (0, 255, 0),      # Green
        'expiry_date': (0, 0, 255),     # Blue
        'quantity': (255, 255, 0)       # Yellow
    }
    
    for item in boxes_data['items']:
        for field, box in item['bounding_boxes'].items():
            x1, y1, x2, y2 = box
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), colors[field], 2)
            cv2.putText(img_with_boxes, field, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[field], 2)
    
    return Image.fromarray(img_with_boxes)

def create_monthly_summary(data):
    """Create monthly summary of expiring drugs"""
    if not data:
        return pd.DataFrame()
    
    records = []
    for item in data['items']:
        records.append({
            'Generic Name': item['generic_name'],
            'Brand Name': item['brand_name'],
            'Expiry Date': item['expiry_date'],
            'Quantity': item['quantity']
        })
    
    df = pd.DataFrame(records)
    df['Expiry Date'] = pd.to_datetime(df['Expiry Date'])
    df['Month-Year'] = df['Expiry Date'].dt.strftime('%Y-%m')
    
    summary = df.groupby(['Month-Year', 'Generic Name', 'Brand Name'])['Quantity'].sum().reset_index()
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
                for idx, item in enumerate(processed_data['items']):
                    edited_data = st.data_editor(
                        pd.DataFrame([{
                            'Generic Name': item['generic_name'],
                            'Brand Name': item['brand_name'],
                            'Expiry Date': item['expiry_date'],
                            'Quantity': item['quantity']
                        }]),
                        key=f"table_{uploaded_file.name}_{idx}"
                    )
                    # Update processed data with edited values
                    processed_data['items'][idx].update({
                        'generic_name': edited_data.iloc[0]['Generic Name'],
                        'brand_name': edited_data.iloc[0]['Brand Name'],
                        'expiry_date': edited_data.iloc[0]['Expiry Date'],
                        'quantity': edited_data.iloc[0]['Quantity']
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