FROM python:3.11
WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY model/autoencoder.h5 /code/autoencoder.h5

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./main.py /code/
COPY ./prediction.py /code/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
