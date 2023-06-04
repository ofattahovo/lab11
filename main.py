from fastapi import FastAPI, UploadFile, File
import uvicorn
import tensorflow as tf
import numpy as np
import base64

# Загрузка обученной модели
model = tf.keras.models.load_model('mnist_cnn.h5')

# Создание экземпляра FastAPI
app = FastAPI()

# Предобработка изображения перед подачей на модель
def preprocess_image(image):
    image = tf.image.decode_image(image, channels=1)
    image = tf.image.resize(image, [28, 28])
    image = tf.cast(image, dtype=tf.float32)
    image = image / 255.0
    image = tf.expand_dims(image, axis=0)
    return image

# Вызов модели для распознавания цифры на предобработанном изображении
def predict_digit(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)

    return predicted_digit.item()

# Определение конечной точки API для распознавания цифры
@app.post("/recognize_digit")
async def recognize_digit(file: UploadFile = File(...)):
    contents = await file.read()
    encoded_image = base64.b64encode(contents).decode("utf-8")
    predicted_digit = predict_digit(contents)
    return {"digit": predicted_digit, "encoded_image": encoded_image}

# Запуск сервера FastAPI с использованием Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000) # curl -X POST -F "file=@3.png" http://127.0.0.1:8000/recognize_digit для отправки запроса

