FROM pytorch/torchserve:latest-gpu

USER root
RUN apt-get update; apt-get install ffmpeg curl vim libsm6 libxext6 -y

# create torchserve configuration file
RUN printf "\ninference_address=http://0.0.0.0:7080" >> /home/model-server/config.properties
RUN printf "\nmanagement_address=http://0.0.0.0:7081" >> /home/model-server/config.properties
RUN printf "\nmetrics_address=http://0.0.0.0:7082" >> /home/model-server/config.properties
COPY ./dockerd-ep.sh /home/model-server/dockerd-ep.sh

RUN chmod +x /home/model-server/dockerd-ep.sh
USER model-server
RUN pip3 install ucr==0.2.16
RUN pip3 install torchserve torch-model-archiver torch-workflow-archiver --upgrade
COPY ./model-archive/ /home/model-server/model-store/
COPY ./otf_message_handler.py /home/model-server/.local/lib/python3.6/site-packages/ts/protocol/otf_message_handler.py

EXPOSE 7080 7081 7082 7070 7071

ENTRYPOINT ["/home/model-server/dockerd-ep.sh"]
CMD ["serve"]
