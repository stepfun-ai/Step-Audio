FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get install -y software-properties-common curl zip unzip git-lfs awscli libssl-dev openssh-server vim \
    && apt-get install -y net-tools iputils-ping iproute2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get install --reinstall ca-certificates && update-ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository -y 'ppa:deadsnakes/ppa' && apt update
RUN apt install python3.10 python3.10-dev python3.10-distutils python3.10-venv -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO- https://bootstrap.pypa.io/get-pip.py | python3.10
RUN ln -s /usr/bin/python3.10 /usr/bin/python
RUN pip uninstall -y Pillow && pip install pillow

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install onnxruntime-gpu==1.17.0  --index-url=https://pkgs.dev.azure.com/onnxruntime/onnxruntime/_packaging/onnxruntime-cuda-12/pypi/simple --force-reinstall --no-deps
