# use python 3.6 base image
FROM python:3.6

# set working directory in the container
WORKDIR /app

# install system dependencies
RUN apt-get update && \
    apt-get install -y libsndfile1 curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

RUN git clone --recursive https://github.com/parlance/ctcdecode.git && \
    cd ctcdecode && pip install .

COPY requirements.txt /app
RUN pip install -r requirements.txt

ENTRYPOINT ["/bin/bash"]