import streamlit as st
import numpy as np
import cv2
from PIL import Image

# --------------------------------------------------
# APP TITLE
# --------------------------------------------------
st.set_page_config(page_title="OpenCV Color Converter", layout="wide")
st.title("🎨 OpenCV Color Space Converter")

st.write("Upload an image and convert it into different OpenCV color spaces.")

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# COLOR CONVERSIONS (SAFE & PRACTICAL)
# --------------------------------------------------
COLOR_CONVERSIONS = {
    "RGB → BGR": cv2.COLOR_RGB2BGR,
    "RGB → GRAY": cv2.COLOR_RGB2GRAY,
    "RGB → HSV": cv2.COLOR_RGB2HSV,
    "RGB → HLS": cv2.COLOR_RGB2HLS,
    "RGB → LAB": cv2.COLOR_RGB2Lab,
    "RGB → LUV": cv2.COLOR_RGB2Luv,
    "RGB → YCrCb": cv2.COLOR_RGB2YCrCb,

    "HSV → RGB": cv2.COLOR_HSV2RGB,
    "LAB → RGB": cv2.COLOR_Lab2RGB,
    "LUV → RGB": cv2.COLOR_Luv2RGB,
    "YCrCb → RGB": cv2.COLOR_YCrCb2RGB,

    "GRAY → RGB": cv2.COLOR_GRAY2RGB,
}

# --------------------------------------------------
# PROCESS IMAGE
# --------------------------------------------------
if uploaded_file is not None:

    # Load image using PIL
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Image (RGB)")
        st.image(img_rgb, use_column_width=True)

    # Select conversion
    conversion_name = st.selectbox(
        "Select Color Conversion",
        list(COLOR_CONVERSIONS.keys())
    )

    conversion_code = COLOR_CONVERSIONS[conversion_name]

    # --------------------------------------------------
    # APPLY CONVERSION SAFELY
    # --------------------------------------------------
    try:
        # Special case: GRAY → RGB requires gray input
        if conversion_name == "GRAY → RGB":
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            converted = cv2.cvtColor(gray, conversion_code)

        # HSV / LAB / LUV / YCrCb reverse conversions
        elif "→ RGB" in conversion_name and "RGB" not in conversion_name.split("→")[0]:
            # Convert RGB to target first
            base_space = conversion_name.split(" → ")[0]
            temp_code = COLOR_CONVERSIONS[f"RGB → {base_space}"]
            temp = cv2.cvtColor(img_rgb, temp_code)
            converted = cv2.cvtColor(temp, conversion_code)

        else:
            converted = cv2.cvtColor(img_rgb, conversion_code)

        with col2:
            st.subheader(f"🎯 Converted Image ({conversion_name})")

            # Grayscale image
            if len(converted.shape) == 2:
                st.image(converted, use_column_width=True)
            else:
                st.image(converted, use_column_width=True)

    except Exception as e:
        st.error(f"❌ Conversion failed: {e}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "🔹 **Project:** OpenCV Color Space Converter  \n"
    "🔹 **Tech:** Python, OpenCV, Streamlit  \n"
    "🔹 **Use Case:** Computer Vision, Image Processing, ML Preprocessing"
)
