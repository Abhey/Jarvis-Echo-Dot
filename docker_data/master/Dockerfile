FROM uselesscoder/alpine_spark_base:v1

MAINTAINER Abhey Rana <abhey.mnnit@gmail.com>
MAINTAINER Jyot Mehta <mailme.jyotmehta@gmail.com>

COPY master.sh /

ENV SPARK_MASTER_PORT 7077
ENV SPARK_MASTER_WEBUI_PORT 8080
ENV SPARK_MASTER_LOG /spark/logs

COPY ./app /app
COPY ./pos_tagger /pos_tagger

RUN pip3 install -r /app/requirements.txt

RUN apk add py-numpy

EXPOSE 8080 7077 6066 2698

CMD ["/bin/bash", "/master.sh"]
