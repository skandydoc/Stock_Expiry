# Pharmacy Inventory Scanner

A Streamlit application that uses Google's Gemini 1.5 Pro Vision model to scan and process pharmacy inventory images, extracting drug information and expiry dates.

## Features

- Upload multiple pharmacy inventory images
- Automatic extraction of:
  - Drug generic names
  - Brand names
  - Expiry dates
  - Quantities
- Visual bounding box detection
- Editable data tables for corrections
- Monthly expiry summary view

## Setup

1. Clone the repository:
```bash
git clone git@github.com:skandydoc/Stock_Expiry.git
cd Stock_Expiry
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

5. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Launch the application using `streamlit run app.py`
2. Upload one or more images of pharmacy inventory
3. The app will process each image and display:
   - Original image with detected text regions
   - Extracted information in an editable table
4. Edit any incorrect information in the tables
5. View the monthly summary of expiring drugs at the bottom of the page

## Requirements

- Python 3.8+
- Google API key with access to Gemini Pro Vision model
- Required Python packages (see requirements.txt)

## Note

Make sure your images are clear and well-lit for optimal text detection. The application supports JPG, JPEG, and PNG formats. 