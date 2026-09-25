# model/

This prototype does not ship a trained leaf-disease classifier — training
one requires a labeled image dataset (e.g. PlantVillage) and a GPU/training
pipeline outside the scope of this generated project.

`app.py`'s `simulate_disease_prediction()` function stands in for real
inference: it deterministically maps each uploaded image to one of five
sample conditions so a demo behaves consistently.

To wire up a real model:
1. Train or download a Keras/TensorFlow model and save it as `plant_model.h5`
   in this folder.
2. In `app.py`, replace the body of `simulate_disease_prediction()` with
   something like:

   ```python
   from tensorflow import keras
   import numpy as np
   from PIL import Image
   import io

   model = keras.models.load_model("model/plant_model.h5")
   CLASS_NAMES = [...]  # match your training labels

   def simulate_disease_prediction(file_bytes):
       img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize((224, 224))
       arr = np.expand_dims(np.array(img) / 255.0, axis=0)
       preds = model.predict(arr)[0]
       idx = int(np.argmax(preds))
       return CLASS_NAMES[idx], round(float(preds[idx]) * 100, 1), TREATMENTS[CLASS_NAMES[idx]]
   ```
3. Add `tensorflow` (or `tensorflow-cpu`) and `Pillow` to `requirements.txt`.
   Note this will significantly increase the deployed image size and may
   exceed Render's free-tier build limits — consider a paid tier or a
   lighter runtime such as `tflite-runtime`.
