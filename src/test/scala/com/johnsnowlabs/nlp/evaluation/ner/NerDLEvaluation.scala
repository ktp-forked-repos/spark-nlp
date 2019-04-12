package com.johnsnowlabs.nlp.evaluation.ner

import java.io.File

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.SparkAccessor.spark.implicits._
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.ner.NerConverter
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.{Benchmark, PipelineModels}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions.{col, udf}
import org.scalatest.FlatSpec


class NerDLEvaluation extends FlatSpec {

  private val trainFile = "/Users/dburbano/JupyterNotebooks/JSL/Assessment/eng.train"
  private val testFile = "/Users/dburbano/JupyterNotebooks/JSL/Assessment/eng.testa"
  evaluateDataSet("Training Dataset", trainFile)
  evaluateDataSet("Testing Dataset", testFile)

  def getNerPipeline: Pipeline = {

    val nerTagger = new NerDLApproach()
      .setInputCols(Array("sentence", "token"))
      .setLabelColumn("label")
      .setOutputCol("ner")
      .setMaxEpochs(10)
      .setRandomSeed(0)
      .setVerbose(2)
      .setEmbeddingsSource("/Users/dburbano/JupyterNotebooks/JSL/Assessment/glove.6B.100d.txt", 100, 2)
      .setExternalDataset(trainFile)

    val converter = new NerConverter()
      .setInputCols(Array("document", "token", "ner"))
      .setOutputCol("ner_span")

    val pipeline = new Pipeline().setStages(
      Array(
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

  def getNerModel(trainDataSet: Dataset[_]): PipelineModel = {
    if (new File("./ner_dl_model").exists()) {
      PipelineModel.load("./ner_dl_model")
    } else {
      val nerPipeline = getNerPipeline
      var model: PipelineModel = null
      Benchmark.measure("Time to train") {
        model = nerPipeline.fit(trainDataSet)
      }
      model.write.overwrite().save("./ner_dl_model")
      model
    }
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
      .filter("ground_truth is not null")
  }

  def getEntitiesLabels(dataSet: Dataset[_], column: String): List[String] = {
    val labels = dataSet.select(dataSet(column)).distinct()
      .rdd.map(row => row.get(0)).collect().toList
      .filter(_ != null)
    labels.asInstanceOf[List[String]]
  }

  def getEvaluationDataSet(dataSet: Dataset[_], labels: List[String]): Dataset[_] = {

    dataSet
      .withColumn("labelIndex", getLabelIndex(labels)(col("ground_truth")))
      .withColumnRenamed("result", "prediction")
      .withColumnRenamed("ground_truth", "label")
      .withColumn("prediction", col("prediction").cast("string"))
      .withColumn("prediction", cleanPrediction(col("prediction")))
      .withColumn("predictionIndex", getLabelIndex(labels)(col("prediction")))
  }

  private def getLabelIndex(labels: List[String]) = udf { label: String =>
      val index = labels.indexOf(label)
      index.toDouble
  }

  private def cleanPrediction = udf { label: String =>
    label.replace("[", "").replace("]", "")
  }

  private def computeAccuracy(dataSet: Dataset[_], labels: List[String]): Unit = {
    val predictionLabelsRDD = dataSet.select("predictionIndex", "labelIndex")
      .map(r => (r.getDouble(0), r.getDouble(1)))
    val metrics = new MulticlassMetrics(predictionLabelsRDD.rdd)
    val accuracy = (metrics.accuracy * 1000).round / 1000.toDouble
    println(s"Accuracy = $accuracy")
    computeAccuracyByEntity(metrics, labels)
  }

  private def computeAccuracyByEntity(metrics: MulticlassMetrics, labels: List[String]): Unit = {
    val predictedLabels = metrics.labels
    predictedLabels.foreach{predictedLabel =>
      val entity = labels(predictedLabel.toInt)
      val precision = (metrics.precision(predictedLabel) * 1000).round / 1000.toDouble
      val recall = (metrics.recall(predictedLabel) * 1000).round / 1000.toDouble
      val f1Score = (metrics.fMeasure(predictedLabel) * 1000).round / 1000.toDouble
      println(s"$entity: Precision = $precision, Recall = $recall, F1-Score = $f1Score")
    }
  }

  private def evaluateDataSet(dataSetType: String, dataSetFile: String): Unit = {
    val trainDataSet = getTrainDataSetWithTokens(dataSetFile)
    val nerModel = getNerModel(trainDataSet)
    var predictionDataSet: Dataset[_] = PipelineModels.dummyDataset
    println(s"Accuracy for $dataSetType")
    Benchmark.measure("Time to transform") {
      predictionDataSet = nerModel.transform(trainDataSet)
        .select("ground_truth", "ner.result")
    }
    Benchmark.measure("Time to show prediction dataset") {
      predictionDataSet.show(5)
    }
    val labels = getEntitiesLabels(predictionDataSet, "ground_truth")
    println("Entities: " + labels)
    val evaluationDataSet = getEvaluationDataSet(predictionDataSet, labels)
    computeAccuracy(evaluationDataSet, labels)
  }

}
