import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_recognize_digit(client):
    # Загрузка изображения
    with open('3.png', 'rb') as image_file:
        image_data = image_file.read()

    # Отправка запроса на распознавание цифры
    response = client.post('/recognize_digit', files={'file': ('3.png', image_data)})

    # Проверка статуса ответа
    assert response.status_code == 200

    # Проверка ожидаемого результата
    expected_digit = 5
    assert response.json()['digit'] == expected_digit

