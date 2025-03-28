FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
# use an older system (18.04) to avoid opencv incompatibility (issue#3524)

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	software-properties-common && \
	add-apt-repository ppa:deadsnakes/ppa && \
	apt-get update && apt-get install -y \
	python3.8 python3.8-dev python3.8-distutils python3-opencv ca-certificates python3-dev git wget sudo ninja-build python3-pip	
RUN ln -sv /usr/bin/python3 /usr/bin/python && \
	ln -sv /usr/bin/pip3 /usr/bin/pip

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.8 get-pip.py && \
    rm get-pip.py
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 2 && \
    update-alternatives --set python /usr/bin/python3.8

# create a non-root user
ARG USER_ID=1000
RUN echo "USER_ID=${USER_ID}" && test -n "${USER_ID}" || (echo "USER_ID is empty!" && exit 1)
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/pip/3.6/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --user tensorboard cmake   # cmake from apt-get is too old
RUN pip install --user torch==1.10 torchvision==0.11.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html


# run detectron2 under user "appuser":
# wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
# python3 demo/demo.py  \
	#--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
	#--input input.jpg --output outputs/ \
	#--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
