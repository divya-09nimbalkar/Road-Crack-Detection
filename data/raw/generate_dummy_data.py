import os
import cv2
import numpy as np

def generate_dummy_images(num_images=10, save_dir="data/raw"):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(num_images):
        # Create a blank road-like image
        img = np.ones((256, 256, 3), dtype=np.uint8) * 200
        
        # Add synthetic "cracks" as black lines
        for _ in range(np.random.randint(1, 5)):
            x1, y1 = np.random.randint(0, 256, 2)
            x2, y2 = np.random.randint(0, 256, 2)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), thickness=2)
        
        cv2.imwrite(os.path.join(save_dir, f"road_{i}.png"), img)

if __name__ == "__main__":
    generate_dummy_images()
    print("✅ Dummy road crack images generated in data/raw/")
