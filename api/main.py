from fastapi import FastAPI, HTTPException

import json
import logging
import os
from fastapi import FastAPI, UploadFile
from prediction import get_prediction, get_model
from fastapi.responses import Response, FileResponse
import io
from PIL import Image

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("LoggerQueries")


DB_URI = os.getenv("DB_URI")

app = FastAPI()

logging.getLogger().setLevel(logging.DEBUG)

model = get_model()

def get_image(image):
    image_bytes: bytes = image
    # media_type here sets the media type of the actual response sent to the client.
    return Response(content=image_bytes, media_type="image/png")

def image_to_byte_array(image: Image) -> bytes:
  # BytesIO is a file-like buffer stored in memory
  imgByteArr = io.BytesIO()
  # image.save expects a file-like as a argument
  image.save(imgByteArr, format=image.format)
  # Turn the BytesIO object back into a bytes object
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr

@app.post("/uploadphoto/", response_class=Response)
async def create_upload_file(file: UploadFile):
    result = get_prediction(file.file.read(), model)
    image = image_to_byte_array(result)
    if result:
        return FileResponse("predict.png")
    # return get_image(image)
    # # return {"result": StreamingResponse(io.BytesIO(result.tobytes()), media_type="image/png")}