package chapter4

import org.apache.spark.SparkConf
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.Vector

import scala.util.Random

object RunRDF_me {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.set("spark.default.parallelism", "4")
    conf.set("spark.master", "local")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    import spark.implicits._
    val dataWithoutHeader = spark.read.option("inferSchema", true).option("header", false).csv("F:\\scala_projects\\aas_mvn\\src\\main\\java\\chapter4\\covtype.data")
    val colNames = Seq("Elevation", "Aspect", "Slope",
      "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
      "Horizontal_Distance_To_Roadways",
      "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
      "Horizontal_Distance_To_Fire_Points") ++ (
      (0 until 4).map(i => s"Wilderness_Area_$i")
      ) ++ (
      (0 until 40).map(i => s"Soil_Type_$i")
      ) ++ Seq("Cover_Type")
    val data: DataFrame = dataWithoutHeader.toDF(colNames: _*).withColumn("Cover_Type", $"Cover_Type".cast("double"))
    val Array(trainData, testData) = data.randomSplit(Array(0.9, 0.1))
    val run: RunRDF = new RunRDF(spark)
//    run.simpleDecisionTree(trainData, testData)
//    run.randomClassifier(trainData, testData)
//    run.evaluate(trainData,testData)
    run.evaluateCategorical(trainData, testData)
  }
}

class RunRDF(private val spark: SparkSession) {

  import spark.implicits._

  def simpleDecisionTree(trainData: DataFrame, testData: DataFrame): Unit = {
    val inputCols: Array[String] = trainData.columns.filter(_ != "Cover_Type")
    val assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("featuresVector")
    val assembleTrainData: DataFrame = assembler.transform(trainData)
    assembleTrainData.select("featuresVector").show(truncate = false)

    val classifier: DecisionTreeClassifier = new DecisionTreeClassifier().setSeed(Random.nextLong()).setLabelCol("Cover_Type").setFeaturesCol("featuresVector").setPredictionCol("prediction")
    val model = classifier.fit(assembleTrainData)
    println(model.toDebugString)

    model.featureImportances.toArray.zip(inputCols).sorted.reverse.foreach(println)
    val prediction = model.transform(assembleTrainData)
    prediction.select("Cover_Type", "prediction", "probability").show(truncate = false)

    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("Cover_Type").setPredictionCol("prediction")
    val accuracy = evaluator.setMetricName("accuracy").evaluate(prediction)
    val f1 = evaluator.setMetricName("f1").evaluate(prediction)
    println(accuracy)
    println(f1)

    val confusionMatrix = prediction.groupBy("Cover_Type").pivot("prediction").count().na.fill(0.0).orderBy("Cover_Type")
    confusionMatrix.show()
  }

  def classProbabilities(data: DataFrame): Array[Double] = {
    val total = data.count()
    data.groupBy("Cover_Type").count().orderBy("Cover_Type").select("count").as[Double].map(_ / total).collect()

  }

  def randomClassifier(trainData: DataFrame, testData: DataFrame): Unit = {
    val trainPriorProb = classProbabilities(trainData)
    val testPriorProb = classProbabilities(testData)
    val accuracy = trainPriorProb.zip(testPriorProb).map {
      case (trainProb, testProb) => trainProb * testProb
    }.sum
    println(accuracy)
  }

  def evaluate(trainData: DataFrame, testData: DataFrame): Unit = {
    val inputCols: Array[String] = trainData.columns.filter(_ != "Cover_Type")
    val assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("featuresVector")
    val classifier = new DecisionTreeClassifier().setSeed(Random.nextLong()).setFeaturesCol("featuresVector").setPredictionCol("prediction").setLabelCol("Cover_Type")

    val pipeline = new Pipeline().setStages(Array(assembler, classifier))
    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.impurity, Seq("entropy", "gini"))
      .addGrid(classifier.maxDepth, Seq(1, 20))
      .addGrid(classifier.maxBins, Seq(40, 300))
      .addGrid(classifier.minInfoGain, Seq(0.0, 0.05))
      .build()
    val multiClassEval = new MulticlassClassificationEvaluator()
      .setLabelCol("Cover_Type")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val validator = new TrainValidationSplit()      //验证器
      .setSeed(Random.nextLong())
      .setEstimator(pipeline)
      .setEvaluator(multiClassEval)
      .setTrainRatio(0.9)
      .setEstimatorParamMaps(paramGrid)

    val validModel = validator.fit(trainData)       //适合于特定数据集的验证器
    val paramsAndMetrics = validModel.validationMetrics
      .zip(validator.getEstimatorParamMaps).sortBy(-_._1)
    paramsAndMetrics.foreach{case (metric,param)=>
      println(metric)
      println(param)
      println()
    }
    val bestModel = validModel.bestModel
    println(bestModel.asInstanceOf[PipelineModel].stages.last.params)
    println(validModel.validationMetrics.max)

    val trainAcc=multiClassEval.evaluate(bestModel.transform(trainData))
    val testAcc=multiClassEval.evaluate(bestModel.transform(testData))

    println(trainAcc)
    println(testAcc)
  }

  def un_encodeOneHot(data:DataFrame):DataFrame={
    val wildernessCols = (0 until 4).map(i => s"Wilderness_Area_$i").toArray
    val soilCols = (0 until 40).map(i => s"Soil_Type_$i").toArray
    val wilderAssembler= new VectorAssembler()
      .setInputCols(wildernessCols)
      .setOutputCol("wilderness")
    val soilAssembler=new VectorAssembler()
      .setInputCols(soilCols)
      .setOutputCol("soil")
    val unhotUDF= udf((vec:Vector) => vec.toArray.indexOf(1.0).toDouble)

    val withWilderness= wilderAssembler.transform(data).drop(wildernessCols: _*).withColumn("wilderness", unhotUDF($"wilderness"))
    soilAssembler.transform(withWilderness).drop(soilCols:_*).withColumn("soil", unhotUDF($"soil"))
  }

  def evaluateCategorical(trainData:DataFrame,testData:DataFrame): Unit ={
    val unencodeTrainData=un_encodeOneHot(trainData)
    val unencodeTestData= un_encodeOneHot(testData)

    val inputCols= unencodeTrainData.columns.filter(_ != "Cover_Type")
    val assembler=new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol("featuresVector")

    val indexer= new VectorIndexer()
      .setMaxCategories(40)
      .setInputCol("featuresVector")
      .setOutputCol("indexedVector")

    val classifier= new DecisionTreeClassifier()
      .setSeed(Random.nextLong())
      .setLabelCol("Cover_Type")
      .setFeaturesCol("indexedVector")
      .setPredictionCol("prediction")

    val paramGrid = new ParamGridBuilder().
      addGrid(classifier.impurity, Seq("gini", "entropy")).
      addGrid(classifier.maxDepth, Seq(1, 20)).
      addGrid(classifier.maxBins, Seq(40, 300)).
      addGrid(classifier.minInfoGain, Seq(0.0, 0.05)).
      build()

    val multiClassEval= new MulticlassClassificationEvaluator()
      .setLabelCol("Cover_Type")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val pipeline=new Pipeline().setStages(Array(assembler, indexer, classifier))

    val validator=new TrainValidationSplit()
      .setSeed(Random.nextLong())
      .setEstimator(pipeline)
      .setEvaluator(multiClassEval)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.9)

    val validatorModel= validator.fit(unencodeTrainData)

    val bestModel = validatorModel.bestModel
    println(bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap)

    val testAcc= multiClassEval.evaluate(bestModel.transform(unencodeTestData))
    println(testAcc)
  }

  def evaluateForest(trainData:DataFrame,testData:DataFrame): Unit ={
    val unencodeTrainData= un_encodeOneHot(trainData)
    val unencodeTestData= un_encodeOneHot(testData)

    val inputCols= unencodeTrainData.columns.filter(_ != "Cover_Type")

    val assembler= new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol("featuresVector")

    val indexer= new VectorIndexer()
      .setInputCol("featuresVector")
      .setOutputCol("indexedVector")
      .setMaxCategories(40)

    val classifier= new RandomForestClassifier()
      .setFeaturesCol("indexedVector")
      .setSeed(Random.nextLong())
      .setMaxBins(300)
      .setMaxDepth(20)
      .setImpurity("entropy")
      .setLabelCol("Cover_Type")
      .setPredictionCol("prediction")

    val pipeline=new Pipeline().setStages(Array(assembler, indexer, classifier))
    val paramGrid=new ParamGridBuilder()
      .addGrid(classifier.minInfoGain,Seq(0.0,0.5))
      .addGrid(classifier.numTrees,Seq(1,10))
      .build()

    val multiClassEval= new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
      .setPredictionCol("prediction")
      .setLabelCol("Cover_Type")

    val validator=new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(multiClassEval)
      .setTrainRatio(0.9)
      .setEstimatorParamMaps(paramGrid)
      .setSeed(Random.nextLong())

    val validatorModels= validator.fit(unencodeTrainData)
    val bestModel = validatorModels.bestModel

    val forestModel=bestModel.asInstanceOf[PipelineModel].stages.last.asInstanceOf[RandomForestClassificationModel]
  }
}