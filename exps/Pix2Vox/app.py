import streamlit as st
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from helpers import *

# --- APP START ---
st.title("2D → 3D Voxel Reconstruction Viewer")

uploaded_images = st.file_uploader(f"Upload images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])
# print(uploaded_images)


# --- DISPLAY ---
if uploaded_images:
    st.subheader("Uploaded Input Views")
    cols = st.columns(len(uploaded_images))
    rendering_images = []

    for i, uploaded_file in enumerate(uploaded_images):
        img = Image.open(uploaded_file)

        cols[i].image(img, caption=f"View {i+1}", use_container_width=True)

        img_np = np.array(img).astype(np.float32) / 255.0

        rendering_images.append(img_np)


    if st.button("Submit for Reconstruction"):
        gv=None
        with st.spinner("Reconstructing..."):
            gv = predict_voxel_from_images(rendering_images)

        fig = voxel_to_plotly(gv)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info(f"Upload images to continue.")
