import streamlit as st
from PIL import Image
from generate_ads import AIImage 

# Streamlit app
st.title("AI Image Generator App")


# Get input data from the user
api_key = st.text_input("API Key:")
text = st.text_input("Text:")
subText = st.text_input("Subtext:")
logo = st.file_uploader("Logo Image (Upload an image file):", type=["png", "jpg", "jpeg"])
description = st.text_input("Description:")
size_width = st.number_input("Image Width:", value=1000)
size_height = st.number_input("Image Height:", value=800)
## image = st.text_input("Image URL:")
colors = st.text_input("Colors (comma-separated):")
font = st.selectbox("Font Type:", ["arial.ttf",])
## target = st.text_input("Target:")
image_type = st.selectbox("Image Type:", ["Instagram", "Youtube"])


# When the user clicks the "Generate Images" button, generate and display images
if st.button("Generate Images"):
    # Create an AIImage instance with the user-provided parameters
    ex = AIImage(
        api_key=api_key,
        text=text,
        subText=subText,
        logo=logo,
        description=description,
        size=(size_width, size_height),
        image="", # image,
        colors=colors,
        font=font,
        target="", # target,
        image_type=image_type
    )

    # Generate images using AIImage
    image_list = ex.generate_images(1)
    imgs = ex.get_designs(6)  # Get 6 random designs from AIImage

    # Display the images
    for idx, img in enumerate(imgs):
        st.image(img, caption=f"Image {idx + 1}", use_column_width=True)

    st.success("Images generated successfully!")
