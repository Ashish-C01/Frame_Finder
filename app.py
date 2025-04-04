import streamlit as st
import chromadb
from chromadb.config import Settings
from transformers import CLIPProcessor, CLIPModel
import cv2
from PIL import Image
import torch
import logging
import uuid
import tempfile
import os
import requests
import json
from dotenv import load_dotenv
import shutil

load_dotenv()
HF_TOKEN = os.getenv('hf_token')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
try:

    @st.cache_resource
    def load_model():
        device = 'cpu'
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14", token=HF_TOKEN)
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14", token=HF_TOKEN)
        model.eval().to(device)
        return processor, model

    @st.cache_resource
    def load_chromadb():
        chroma_client = chromadb.PersistentClient(
            path='Data', settings=Settings(anonymized_telemetry=False))
        collection = chroma_client.get_or_create_collection(name='images')
        return chroma_client, collection

    def resize_image(image_path, size=(224, 224)):
        if isinstance(image_path, str):
            img = Image.open(image_path).convert("RGB")
        else:
            img = Image.open(image_path).convert("RGB")
        img_resized = img.resize(size, Image.LANCZOS)
        return img_resized

    def get_image_embedding(image, model, preprocess, device='cpu'):
        image = Image.open(f'{image}').convert('RGB')
        input_tensor = preprocess(images=[image], return_tensors='pt')[
            'pixel_values'].to(device)
        with torch.no_grad():
            embedding = model.get_image_features(
                pixel_values=input_tensor)

        return torch.nn.functional.normalize(embedding, p=2, dim=1)

    def extract_frames(v_path, frame_interval=30):
        cap = cv2.VideoCapture(v_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        total_seconds = frame_count//frame_rate
        frame_idx = 0
        saved_frames = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                unique_image_id = str(uuid.uuid4())
                frame_name = f"{temp_dir}/frame_{unique_image_id}_{saved_frames}.jpg"
                cv2.imwrite(frame_name, frame)

                saved_frames += 1
            frame_idx += 1
        cap.release()

    def insert_into_db(collection, dir):
        embedding_list = []
        file_names = []
        ids = []
        with st.status("Generating embedding... ⏳", expanded=True) as status:
            for i in os.listdir(dir):
                embedding = get_image_embedding(
                    f"{dir}/{i}", model, processor)
                embedding_list.append(
                    embedding.squeeze(0).numpy().tolist())
                file_names.append(
                    {'path': f"{dir}/{i}", 'type': 'photo'})
                unique_id = str(uuid.uuid4())
                ids.append(unique_id)
            status.update(label="Embedding generation complete",
                          state="complete")

        collection.add(
            embeddings=embedding_list,
            ids=ids,
            metadatas=file_names
        )
        logger.info("Data inserted into DB")

    processor, model = load_model()
    logger.info("Model and processor loaded")
    client, collection = load_chromadb()
    logger.info("ChromaDB loaded")
    logger.info(
        f"Connected to ChromaDB collection images with {collection.count()} items")

    temp_dir = 'temp_folder'
    if 'cleaned_temp' not in st.session_state:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        st.session_state.cleaned_temp = True
        results = collection.get(include=["metadatas"])
        ids_to_delete = [
            _id for _id, metadata in zip(results["ids"], results['metadatas']) if metadata.get("path", "").startswith("temp")
        ]
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)

    st.title("Extract frames from video using text")
    # Upload section
    st.sidebar.subheader("Upload video")
    video_file = st.sidebar.file_uploader(
        "Upload videos", type=["mp4", "webm", "avi", "mov"], accept_multiple_files=False
    )
    num_images = st.sidebar.slider(
        "Number of images to  be shown", min_value=1, max_value=10, value=3)
    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
            tmpfile.write(video_file.read())
            video_path = tmpfile.name
        st.video(video_path)
        st.sidebar.subheader("Add uploaded videos to collection")
        if st.sidebar.button("Add uploaded video"):
            extract_frames(video_path)
            insert_into_db(collection, temp_dir)
    else:
        video_path = 'Videos/Video.mp4'
        st.video(video_path)
        st.write(
            f"Video credits: https://www.kaggle.com/datasets/icebearisin/raw-skates")

    st.write("Enter the description of image to be  extracted from the video")
    text_input = st.text_input("Description", "Flying Skater")
    if st.button("Search"):
        if text_input.strip():
            params = {'text': text_input.strip()}
            response = requests.get(
                'https://ashish-001-text-embedding-api.hf.space/embedding', params=params)
            if response.status_code == 200:
                logger.info("Embedding returned by API successfully")
                data = json.loads(response.content)
                embedding = data['embedding']
                results = collection.query(
                    query_embeddings=[embedding],
                    n_results=num_images
                )
                images = [results['metadatas'][0][i]['path']
                          for i in range(len(results['metadatas'][0]))]
                distances = [results['distances'][0][i]
                             for i in range(len(results['metadatas'][0]))]
                if images:
                    cols_per_row = 3
                    rows = (len(images)+cols_per_row-1)//cols_per_row
                    for row in range(rows):
                        cols = st.columns(cols_per_row)
                        for col_idx, col in enumerate(cols):
                            img_idx = row*cols_per_row+col_idx
                            if img_idx < len(images):
                                resized_img = resize_image(
                                    images[img_idx])
                                col.image(resized_img,
                                          caption=f"Image {img_idx+1}", use_container_width=True)
                else:
                    st.write("No image found")
            else:
                st.write("Please try again later")
                logger.info(f"status code {response.status_code} returned")
        else:
            st.write("Please enter text in the text area")

except Exception as e:
    logger.exception(f"Exception occured, {e}")
