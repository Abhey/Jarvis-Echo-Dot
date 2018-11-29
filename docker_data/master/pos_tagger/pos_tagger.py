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


spark = SparkSession.builder \
    .appName("ner")\
    .master("local[1]")\
    .config("spark.driver.memory", "8G")\
    .config("spark.driver.maxResultSize", "2G") \
    .config("spark.driver.extraClassPath", "lib/sparknlp.jar")\
    .config("spark.kryoserializer.buffer.max", "500m")\
    .getOrCreate()

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["document"])\
    .setOutputCol("token")

posTagger = PerceptronApproach()\
    .setIterations(5)\
    .setInputCols(["token", "document"])\
    .setOutputCol("pos")\
    .setCorpus("/pos_tagger/resources/anc-pos-corpus-small/", "|")

finisher = Finisher() \
    .setInputCols(["pos"]) \
    .setIncludeMetadata(True)

pipeline = Pipeline(
    stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        finisher
    ])

data = spark. \
    read. \
    parquet("/pos_tagger/resources/sentiment.parquet"). \
    limit(1000)

data.cache()
# data.count()
# data.show()

model = pipeline.fit(data)

with open('/pos_tagger/resources/channel.txt', 'r') as content_file:
    content = content_file.read()

test_data = spark.sparkContext.parallelize([[content]]).toDF().toDF("text")
test_data.show()

res_data = model.transform(test_data)
res_data.show()

pos = res_data.collect()
result = pos[0].finished_pos

try:
    output_file = open("/pos_tagger/resources/output.txt", "w")
    output_file.write(result)
except Exception as e:
        print("An excetion occurred while catching another excetion.")

model.write().overwrite().save("/pos_tagger/resources/pipeline_trained/")

load_model = PipelineModel.read().load("/pos_tagger/resources/pipeline_trained/")
# load_model.transform(data).show()
