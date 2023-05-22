from fastapi import FastAPI, HTTPException

import json
import logging
import os
from fastapi import FastAPI, UploadFile
from prediction import get_prediction, get_model

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("LoggerQueries")


DB_URI = os.getenv("DB_URI")

app = FastAPI()

logging.getLogger().setLevel(logging.DEBUG)

model = get_model()

@app.post("/uploadphoto/")
async def create_upload_file(file: UploadFile):
    print(type(file.file))
    result = get_prediction(file.file.read(), model)
    return {"result": result}