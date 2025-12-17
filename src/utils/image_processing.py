import cv2
import numpy as np
import pandas as pd
from PIL import Image

def process_ecg_image(uploaded_file):
    # Read image as grayscale
    img_pil = Image.open(uploaded_file).convert("L")
    img_np = np.array(img_pil)

    # Resize for consistent processing
    img_resized = cv2.resize(img_np, (1200, 800))  # width x height

    # Divide into 12 leads (3 rows x 4 columns)
    h, w = img_resized.shape
    lead_height = h // 3
    lead_width = w // 4

    leads = []
    signals = []
    for row in range(3):
        for col in range(4):
            x1 = col * lead_width
            y1 = row * lead_height
            lead_img = img_resized[y1:y1+lead_height, x1:x1+lead_width]
            leads.append(lead_img)

            # Extract 1D signal using horizontal line profile from middle row
            mid_row = lead_img.shape[0] // 2
            signal = 255 - lead_img[mid_row, :]  # invert to make peaks go up
            signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-8)  # normalize
            signals.append(signal)

    # Combine all signals into DataFrame
    max_len = max(len(sig) for sig in signals)
    signal_data = {f"Lead_{i+1}": np.pad(sig, (0, max_len - len(sig)), mode='constant') for i, sig in enumerate(signals)}
    signal_df = pd.DataFrame(signal_data)

    return signal_df, img_resized, leads
