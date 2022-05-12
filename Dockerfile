FROM gocv/opencv:latest

WORKDIR /usr/src/app

RUN apt-get update \
    && apt-get install python3-pip -y \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN  pip3 install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

CMD [ "python3", "./server.py" ]
