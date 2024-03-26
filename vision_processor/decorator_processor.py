import pickle
import base64
import os
from io import BytesIO
from PIL import Image


def save_to_bytes(func):
    def wrapper(self, data):
        bytes_data = pickle.dumps(data)
        return func(self, bytes_data)

    return wrapper


def save_to_one_file(func):
    def wrapper(self, data, filename):
        with open(filename, "wb") as file:
            func(self, data, file)

    return wrapper


def save_to_file(func):
    def wrapper(self, data, filename, quality):
        os.makedirs(filename, exist_ok=True)
        for i, image_data in enumerate(data):
            file_path = os.path.join(filename, f"{i+1}.jpg")

            func(self, image_data, file_path, quality)

    return wrapper


def save_to_base64(func):
    def wrapper(self, data, quality=95):
        rlt = Image.fromarray(data)

        with BytesIO() as byte_output:
            rlt.save(byte_output, format="JPEG", quality=quality)
            byte_output.seek(0)
            byte_data = byte_output.read()
        base64_data = base64.b64encode(byte_data).decode("utf-8")
        return func(self, base64_data)

    return wrapper
