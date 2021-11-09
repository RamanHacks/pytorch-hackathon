FROM pytorch/torchserve:latest-gpu

USER root
RUN apt-get update; apt-get install ffmpeg curl vim libsm6 libxext6 -y

USER model-server
RUN pip3 install ucr==0.2.16
RUN pip3 install torchserve torch-model-archiver torch-workflow-archiver --upgrade
RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# COPY ./config.properties /home/model-server/config.properties
# COPY ./index_funsd.json /home/model-server/index_funsd.json
# COPY ./index_sroie.json /home/model-server/index_sroie.json
# COPY ./dockerd-ep.sh /home/model-server/dockerd-ep.sh
# COPY ./jit-models /home/model-server/
USER root
COPY . /home/model-server/
RUN chmod 777 -R /home/model-server/
USER model-server

EXPOSE 7081 7080 7082

ENTRYPOINT ["/home/model-server/dockerd-ep.sh"]
CMD ["serve"]
