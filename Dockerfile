FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

WORKDIR /app

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN apt update

COPY ./src ./src
COPY ./app.py ./app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
