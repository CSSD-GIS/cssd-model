FROM pytorch/pytorch:latest

WORKDIR /usr/src/app

RUN apt-get update
RUN apt-get install libgl1-mesa-glx -y

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt 

COPY . .

CMD [ "python3", "./server.py" ]
