LOW-RES IMAGE CLASSIFICATION
import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab import files
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
# Load Pretrained Model
model = MobileNetV2(weights='imagenet')
IMG_SIZE = 224
def create_low_resolution(img):
    small = cv2.resize(img, (IMG_SIZE//2, IMG_SIZE//2),
                       interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (IMG_SIZE, IMG_SIZE),
                          interpolation=cv2.INTER_CUBIC)
    return restored
# Upload PNG Image
print("Upload your PNG image")
uploaded = files.upload()
img_name = list(uploaded.keys())[0]
# Read Image
img = cv2.imread(img_name)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# Apply Low Resolution + Super Resolution
low_res_img = create_low_resolution(img)
# Prepare for model
img_array = image.img_to_array(low_res_img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Prediction
predictions = model.predict(img_array)
decoded = decode_predictions(predictions, top=3)[0]
# Show Image
plt.imshow(low_res_img)
plt.axis("off")
plt.show()
# Print Results
print("Top Predictions:")
for i, pred in enumerate(decoded):
    print(f"{i+1}. {pred[1]} - {round(pred[2]*100,2)}%")





