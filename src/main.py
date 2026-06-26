import os
import cv2
import src.data_pipeline as data_pipeline
import src.model as model
import src.train as train
import src.evaluate as evaluate



def run():
    print("🚀 Road Crack Detection Project")
    data_pipeline.prepare_data()
    net = model.build_model()
    train.train_model(net)
    evaluate.evaluate_model(net)

    # Load one of the generated dummy images
    img_path = "data/raw/road_0.png"
    if not os.path.exists(img_path):
        print("⚠️ No image found, generating dummy data...")
        from data.raw.generate_dummy_data import generate_dummy_images
        generate_dummy_images()
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load {img_path}")
    print(f"✅ Loaded image {img_path} with shape {img.shape}")

if __name__ == "__main__":
    run()
