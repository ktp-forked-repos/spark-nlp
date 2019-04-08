package com.johnsnowlabs.nlp.evaluation.ner

import java.sql.Struct

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType, DocumentAssembler, Finisher}
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.ner.NerConverter
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.avro.generic.GenericData.StringType
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{Column, Dataset, Row}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, typedLit, udf}
import org.apache.spark.sql.types.{ArrayType, MetadataBuilder, StructField, StructType}
import org.scalatest.FlatSpec

import scala.collection.mutable.ArrayBuffer


class NerDLEvaluation extends FlatSpec {

  private val trainFile = "/Users/dburbano/JupyterNotebooks/JSL/Assessment/eng.train"
  private val trainDataSet = getTrainDataSetWithTokens(trainFile)
  trainDataSet.show(5, false)
  private val nerModel = getNerModel
  private val predictionDataSet = nerModel.transform(trainDataSet)
    .select("ground_truth", "ner.result")
  predictionDataSet.show(5, false)
  predictionDataSet.printSchema()
  private lazy val labels = getEntitiesLabels(predictionDataSet, "ground_truth")
  println(labels)
  private val evaluationDataSet = getEvaluationDataSet(predictionDataSet)
  evaluationDataSet.show(5, false)
  evaluationDataSet.printSchema()
  evaluationDataSet.withColumn("predictionIndex", customNer(col("prediction")))
  evaluationDataSet.show()

  def getNerPipeline: Pipeline = {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")
      .setTrimAndClearNewLines(false)

    val sentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")
      .setPrefixPattern("\\A([^\\s\\p{L}$\\.']*)")

    val nerTagger = new NerDLApproach()
      .setInputCols(Array("sentence", "token"))
      .setLabelColumn("label")
      .setOutputCol("ner")
      .setMaxEpochs(1)
      .setRandomSeed(0)
      .setVerbose(2)
      .setEmbeddingsSource("glove.6B.100d.txt", 100, 2)

    val converter = new NerConverter()
      .setInputCols(Array("document", "token", "ner"))
      .setOutputCol("ner_span")

    val pipeline = new Pipeline().setStages(
      Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        nerTagger,
        converter))

    pipeline

  }

  def getTokenPipeline: Pipeline = {
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")
      .setTrimAndClearNewLines(false)

    val sentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")
      .setPrefixPattern("\\A([^\\s\\p{L}$\\.']*)")
      .setIncludeDefaults(false)

    val pipeline = new Pipeline().setStages(
      Array(
        documentAssembler,
        sentenceDetector,
        tokenizer))

    pipeline

  }

  def getNerModel: PipelineModel = {
    PipelineModel.load("/Users/dburbano/PycharmProjects/TestingJSL/model/ner_dl_model")
  }

  def getTrainDataSetWithTokens(pathDataSet: String): Dataset[_] = {
    var trainDataSet = ResourceHelper.spark.read.option("delimiter", " ").csv(pathDataSet)
      .withColumnRenamed("_c0", "text")
      .withColumnRenamed("_c3", "ground_truth")
      .limit(100)

    trainDataSet = trainDataSet.select("text", "ground_truth")
    trainDataSet = trainDataSet.filter("text != '-DOCSTART-'")
    val pipeline = getTokenPipeline
    pipeline.fit(trainDataSet).transform(trainDataSet)
  }

  def getEntitiesLabels(dataSet: Dataset[_], column: String): List[String] = {
    val labels = dataSet.select(dataSet(column)).distinct()
      .rdd.map(row => row.get(0)).collect().toList
      .filter(_ != null)
    labels.asInstanceOf[List[String]]
  }

  def getEvaluationDataSet(dataSet: Dataset[_]): Dataset[_] = {
    //  predictionDataSet = predictionDataSet.withColumn("labelIndex",
    //    when(col("ground_truth") =!= "", labels.indexOf(col("ground_truth"))).otherwise(0))
    //private val getLabelIndexUDF = udf{label: String => label.toLowerCase}
    //private val getLabelIndexUDF = udf(toLower _)

    dataSet
      .withColumn("labelIndex", getLabelIndex(labels)(col("ground_truth")))
      .withColumnRenamed("result", "prediction")
      //.withColumn("prediction", col("result").cast("string"))
      //.withColumn("prediction", customNer(col("result")))
  }

  private def getLabelIndex(labels: List[String]) = udf { label: String =>
    val index = labels.indexOf(label)
    index.toString
  }

  private def customNer = udf { ner: Array[String] =>
    ner.head
  }


}
