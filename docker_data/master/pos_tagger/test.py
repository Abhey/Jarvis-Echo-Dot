import os
import sys
sys.path.append('../../')

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel, Pipeline

from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *

import time
import zipfile

load_model = PipelineModel.read().load("/pos_tagger/resources/pipeline_trained/")

import socket

spark = SparkSession.builder \
    .appName("ner")\
    .master("local[1]")\
    .config("spark.driver.memory", "8G")\
    .config("spark.driver.maxResultSize", "2G") \
    .config("spark.driver.extraClassPath", "lib/sparknlp.jar")\
    .config("spark.kryoserializer.buffer.max", "500m")\
    .getOrCreate()

s = socket.socket()

port = 2698
s.bind(('', port))

s.listen(5)

while True:
	c, addr=s.accept()
	txt=c.recv(1024)
	test_data=spark.sparkContext.parallelize([[txt]]).toDF().toDF("text")
	res_data=load_model.transform(test_data)
	pos=res_data.collect()
	result=pos[0].finished_pos
	print(result)
	c.send(result)
	c.close()
