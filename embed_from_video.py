from transformers import CLIPProcessor, CLIPModel
from chromadb.config import Settings
import chromadb
from PIL import Image
import torch
import cv2
import os
import uuid

try:
    chroma_client = chromadb.PersistentClient(
        path='Data', settings=Settings(anonymized_telemetry=False))
    collection = chroma_client.get_or_create_collection(name='images')
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14", local_files_only=True)
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14", local_files_only=True)
    temp_dir = 'frames'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)

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
        for i in os.listdir(dir):
            embedding = get_image_embedding(
                f"{dir}/{i}", model, processor)
            embedding_list.append(
                embedding.squeeze(0).numpy().tolist())
            file_names.append(
                {'path': f"{dir}/{i}", 'type': 'photo'})
            unique_id = str(uuid.uuid4())
            ids.append(unique_id)

        collection.add(
            embeddings=embedding_list,
            ids=ids,
            metadatas=file_names
        )

    def main(video_file_path, t_dir):
        extract_frames(video_file_path)
        insert_into_db(collection, t_dir)

    if __name__ == "__main__":
        video_path = os.path.join('Videos', 'Video.mp4')
        main(video_path, temp_dir)

except Exception as e:
    print(f"Exception occured, {e}")
