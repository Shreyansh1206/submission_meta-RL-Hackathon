FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    sumo \
    sumo-tools \
    sumo-doc \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV SUMO_HOME="/usr/share/sumo"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app

CMD ["python3", "inference.py"]