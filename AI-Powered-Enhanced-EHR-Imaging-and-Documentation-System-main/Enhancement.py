import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D  # type: ignore
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

# ✅ FIXED PATH ISSUE
base_dir = os.path.dirname(__file__)
processed_folder = os.path.join(base_dir, "Xray_processed")

# 1️⃣ LOAD PROCESSED IMAGES
image_files = [os.path.join(processed_folder, f) for f in os.listdir(processed_folder)
               if f.lower().endswith((".jpeg", ".jpg", ".png"))]

images = []
for file in image_files:
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (256, 256))
        img = img.astype("float32") / 255.0
        images.append(img)

images = np.array(images)
images = np.expand_dims(images, axis=-1)
print(f"✅ Loaded {len(images)} images from {processed_folder}")

# 2️⃣ ADD NOISE
noise_factor = 0.2
x_noisy = np.clip(images + noise_factor * np.random.normal(0, 1, images.shape), 0., 1.)

# 3️⃣ TRAIN / TEST SPLIT
split = int(0.8 * len(images))
x_train, x_test = x_noisy[:split], x_noisy[split:]
y_train, y_test = images[:split], images[split:]
print(f"🧠 Train: {len(x_train)}, Test: {len(x_test)}")

# 4️⃣ AUTOENCODER MODEL
input_img = Input(shape=(256, 256, 1))
x = Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same')(x)

x = Conv2D(32, (3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# 5️⃣ TRAIN
autoencoder.fit(
    x_train, y_train,
    epochs=10,
    batch_size=8,
    shuffle=True,
    validation_data=(x_test, y_test),
    verbose=1
)

# 6️⃣ ENHANCE TEST IMAGES
enhanced = autoencoder.predict(x_test)
print("✅ Image enhancement completed.")

# 7️⃣ SAVE ENHANCED IMAGES
enhanced_folder = os.path.join(base_dir, "Xray_enhanced")
os.makedirs(enhanced_folder, exist_ok=True)
for i in range(len(enhanced)):
    save_path = os.path.join(enhanced_folder, f"enhanced_{i}.png")
    cv2.imwrite(save_path, (enhanced[i].squeeze() * 255).astype(np.uint8))
print(f"✅ Enhanced images saved in '{enhanced_folder}'")

# 8️⃣ METRICS
num_samples = min(len(y_test), len(enhanced))
ssim_vals = [ssim(y_test[i].squeeze(), enhanced[i].squeeze(), data_range=1.0)
             for i in range(num_samples)]
psnr_vals = [psnr(y_test[i].squeeze(), enhanced[i].squeeze(), data_range=1.0)
             for i in range(num_samples)]
print(f"📊 Avg SSIM: {np.mean(ssim_vals):.4f}, Avg PSNR: {np.mean(psnr_vals):.2f} dB")

# 9️⃣ VISUALIZE
num_show = min(5, len(y_test))
for i in range(num_show):
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(x_test[i].squeeze(), cmap='gray')
    plt.title("Before Enhancement (Noisy)")
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(enhanced[i].squeeze(), cmap='gray')
    plt.title("After Enhancement (GenAI)")
    plt.axis('off')
    
    plt.suptitle(f"X-ray {i+1}: Before vs After", fontsize=12)
    plt.tight_layout()
    plt.show()

print("\n🎯 Completed – Before/After visual comparison & metrics calculated successfully!")
