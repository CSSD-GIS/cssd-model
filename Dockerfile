FROM pytorch/pytorch:latest

WORKDIR /app

RUN apt-get update \
    && apt-get install libgl1-mesa-glx libglib2.0-dev -y \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt .
RUN pip3 install -r requirements.txt

EXPOSE 9001

CMD [ "python3", "./server.py" ]
