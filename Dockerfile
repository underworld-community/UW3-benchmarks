ARG BASE_IMAGE=julesg/underworld3:0.9-x86_64
FROM ${BASE_IMAGE}

# Make sure the contents of our repo are in /home/jovyan
COPY . /home/jovyan
USER root
RUN chown -R ${NB_UID} /home/jovyan
USER ${NB_UID}
