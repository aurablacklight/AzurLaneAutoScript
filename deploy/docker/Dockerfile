# docker build -t hgjazhgj/alas:latest .
# docker run -v ${PWD}:/app/AzurLaneAutoScript -p 22267:22267 --name alas -it --rm hgjazhgj/alas
#
# Runs natively on both amd64 and arm64.
# OCR (mxnet) is handled by the separate Dockerfile.ocr container.

FROM python:3.9-slim

WORKDIR /app/AzurLaneAutoScript

COPY requirements.txt /tmp/requirements.txt

RUN apt update \
 && apt install -y git adb libgomp1 openssh-client gcc g++ python3-dev libev-dev libffi-dev autoconf automake libtool make pkg-config \
 && apt install -y libavformat-dev libavcodec-dev libavdevice-dev \
 && apt install -y libavutil-dev libswscale-dev libswresample-dev libavfilter-dev \
 && git config --global --add safe.directory '*' \
 && pip install --upgrade pip \
 && pip install --no-deps cnocr==1.2.2 \
 && pip install Cython==0.29.37 \
 && pip install av==10.0.0 --no-build-isolation \
 && GEVENT_NO_CARES_BUILD=1 pip install gevent \
 && pip install -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt \
 && rm -rf ~/.cache/pip

CMD python gui.py
