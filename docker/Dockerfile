FROM nvcr.io/nvidia/pytorch:21.05-py3

# The commented lines are meant for internal use by KCL BMEIS staff
#ARG USER_ID
#ARG GROUP_ID
#ARG USER

ENV TZ=Europe/London

#RUN addgroup --gid $GROUP_ID $USER
#RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER

# This is required for cases where the sqsh file is run without internet connection
RUN rm -rf ~/.cache
RUN rm -rf /root/.cache
RUN mkdir /cache_dir
RUN ln -s /cache_dir ~/.cache
RUN ln -s /cache_dir /root/.cache
ENV XDG_CACHE_HOME=/cache_dir
ENV TORCH_HOME=/cache_dir
ENV MPLCONFIGDIR=/cache_dir

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y sudo
RUN pip3 install --upgrade pip
RUN apt-get install -y ffmpeg

RUN pip install -U --no-cache-dir git+https://github.com/idiap/fast-transformers.git@c99d771cdff096ce44336e06d9fcf2fe163b7626
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt

# This is required for cases where the sqsh file is run without internet connection
RUN python3 -c "import lpips;lpips.LPIPS(net='alex')"

ENV BUILD_MONAI=1
RUN wget https://github.com/Project-MONAI/MONAI/archive/0aa936f87a694a66d54e514ec823a37e999be862.zip && \
    unzip 0aa936f87a694a66d54e514ec823a37e999be862.zip && \
    rm 0aa936f87a694a66d54e514ec823a37e999be862.zip && \
    cd MONAI-0aa936f87a694a66d54e514ec823a37e999be862 && \
    python3 setup.py develop

#USER $USER

#RUN source /home/$USER/.bashrc
#RUN source /home/$USER/.profile

ENTRYPOINT ["/bin/bash"]
