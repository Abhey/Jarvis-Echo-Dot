FROM uselesscoder/alpine_spark_base:v1

MAINTAINER Abhey Rana <abhey.mnnit@gmail.com>
MAINTAINER Jyot Mehta <mailme.jyotmehta@gmail.com>

COPY worker.sh /

ENV SPARK_WORKER_WEBUI_PORT 8081
ENV SPARK_WORKER_LOG /spark/logs
ENV SPARK_MASTER "spark://spark-master:7077"

COPY ./app /app
COPY ./pos_tagger /pos_tagger

RUN pip3 install -r /app/requirements.txt

RUN apk add py-numpy

EXPOSE 8081

CMD ["/bin/bash", "/worker.sh"]
