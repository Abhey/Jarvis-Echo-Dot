FROM alpine_base

MAINTAINER Abhey Rana <abhey.mnnit@gmail.com>
MAINTAINER Jyot Mehta <mailme.jyotmehta@gmail.com>

COPY template.sh /

# Copy the source code
COPY ./app /app
COPY ./pos_tagger /pos_tagger

RUN pip3 install -r /app/requirements.txt

RUN apk add py-numpy

EXPOSE 80

CMD ["/bin/bash", "/template.sh"]
CMD ["python3", "/app/app.py"]