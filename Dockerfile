ARG BASE=python:3
FROM $BASE

COPY requirements.txt /tmp/requirements.txt

# install opencv (long running...)
RUN apt-get update && \
    apt-get install --no-install-recommends -y python3-opencv && \
    pip install -r /tmp/requirements.txt

RUN echo "X11Forwarding yes" >> /etc/ssh/ssh_config
RUN echo "X11UseLocalhost no" >> /etc/ssh/ssh_config

# setup the user
ARG USER=masked
RUN groupadd -g 1000 $USER
RUN useradd -d /home/$USER -s /bin/bash -m $USER -u 1000 -g 1000 \
    && echo $USER:ubuntu | chpasswd \
    && adduser $USER sudo

# copy the app src
COPY ./src /src

# setup stuff
WORKDIR /home/$USER
USER $USER
ENV HOME /home/$USER

ENTRYPOINT ["python", "/src/mask-editor.py"]
