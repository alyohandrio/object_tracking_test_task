FROM python:3.10

COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN mkdir /detector
WORKDIR /detector

COPY . .

ENTRYPOINT ["python3", "main.py", "-c", "configs/config.json"]
