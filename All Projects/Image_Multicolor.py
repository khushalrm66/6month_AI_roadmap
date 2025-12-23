import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Image Processor", layout="wide")

# Title
st.title("Multi-Color Channel Image Visualizer")

# ================= IMAGE INPUT =================
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    elephant = Image.open(uploaded_file).convert("RGB")

    # Show original image
    st.image(elephant, caption="Original Image", use_container_width=True)

    # Convert to NumPy
    elephant_np = np.array(elephant)
    R, G, B = elephant_np[:, :, 0], elephant_np[:, :, 1], elephant_np[:, :, 2]

    # Create channel images
    red_img = np.zeros_like(elephant_np)
    green_img = np.zeros_like(elephant_np)
    blue_img = np.zeros_like(elephant_np)

    red_img[:, :, 0] = R
    green_img[:, :, 1] = G
    blue_img[:, :, 2] = B

    # ================= RGB CHANNELS =================
    st.subheader("RGB Channel Visualization")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(red_img, caption="Red Channel", use_container_width=True)

    with col2:
        st.image(green_img, caption="Green Channel", use_container_width=True)

    with col3:
        st.image(blue_img, caption="Blue Channel", use_container_width=True)

    # ================= COLORMAP =================
    st.subheader("Colormapped Grayscale Image")

    colormap = st.selectbox(
        "Choose a Matplotlib colormap",
        ["viridis", "plasma", "inferno", "magma", "cividis", "hot", "cool", "gray"]
    )

    elephant_gray = elephant.convert("L")
    elephant_gray_np = np.array(elephant_gray)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(elephant_gray_np, cmap=colormap)
    ax.axis("off")

    st.pyplot(fig)

else:
    st.info(" Please upload an image to continue")
