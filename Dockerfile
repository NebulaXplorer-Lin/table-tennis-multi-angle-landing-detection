FROM sport-base:v1

WORKDIR /participant

COPY . /participant/

ENTRYPOINT ["bash", "/participant/run_all.sh"]