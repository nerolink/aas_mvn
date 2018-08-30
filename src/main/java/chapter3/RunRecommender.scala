package chapter3

import org.apache.spark.SparkConf
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.recommendation.ALS

import scala.collection.Map
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object RunRecommender {


  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.set("spark.default.parallelism", "1")
    conf.set("executor-cores", "16")
    conf.set("executor-memory","6g")
    conf.set("driver-memory","8g")
    val spark = SparkSession.builder().master("local").config(conf).appName("Run Recommender").getOrCreate()
    val rawUserArtistData = spark.read.textFile("G:\\aas_data\\chapter2\\user_artist_data.txt").repartition(16)
    val rawArtistData = spark.read.textFile("G:\\aas_data\\chapter2\\extract\\artist_data.txt")
    val rawArtistAlias = spark.read.textFile("G:\\aas_data\\chapter2\\extract\\artist_alias.txt")

    val rr: RunRecommender = new RunRecommender(spark)
    rr.model(rawUserArtistData, rawArtistData, rawArtistAlias)
  }

  class RunRecommender(private val spark: SparkSession) {

    import spark.implicits._

    def buildArtistAlias(rawArtistAlias: Dataset[String]): Map[Int, Int] = {
      rawArtistAlias.flatMap { line =>
        val Array(artist, alias) = line.split('\t')
        if (artist.isEmpty) {
          None
        } else {
          Some((artist.toInt, alias.toInt))
        }
      }.collect().toMap
    }

    def buildArtistById(rawArtistData: Dataset[String]): DataFrame = {
      rawArtistData.flatMap { line =>
        val (id, name) = line.span(_ != "\t")
        if (name.isEmpty()) {
          None
        } else {
          try {
            Some(id.toInt, name.trim())
          } catch {
            case _: NumberFormatException => None
          }
        }
      }.toDF("id", "name")
    }

    def buildUserArtist(rawUserArtistData: Dataset[String]): DataFrame = {
      rawUserArtistData.map { line =>
        val Array(user, artist, _*) = line.split(' ')
        (user.toInt, artist.toInt)
      }.toDF("user", "artist")
    }

    def buildCounts(rawUserArtistData: Dataset[String], bArtistAlias: Broadcast[Map[Int, Int]], spark: SparkSession): DataFrame = {
      import spark.implicits._
      rawUserArtistData.map { line =>
        val Array(user, artist, count) = line.split(" ").map(_.toInt)
        val finalArtist = bArtistAlias.value.getOrElse(artist, artist)
        (user, finalArtist, count)
      }.toDF("user", "artist", "count")
    }

    def model(rawUserArtistData: Dataset[String], rawArtistData: Dataset[String], rawArtistAlias: Dataset[String]): Unit = {
      val userArtistData = buildUserArtist(rawUserArtistData)
      val artistData = buildArtistById(rawArtistData)
      val artistAlias = buildArtistAlias(rawArtistAlias)
      val bArtistAlias = spark.sparkContext.broadcast(artistAlias)
      val train = buildCounts(rawUserArtistData, bArtistAlias, spark)

      val model = new ALS().
        setSeed(Random.nextLong()).
        setImplicitPrefs(true).
        setRank(10).
        setRegParam(0.01).
        setAlpha(1.0).
        setMaxIter(5).
        setUserCol("user").
        setItemCol("artist").
        setRatingCol("count").
        setPredictionCol("prediction").
        fit(train)
      train.unpersist()
      model.userFactors.select("features").show(truncate = false)

      val userID = 2093760

      @transient
      val rec_items = model.recommendForAllItems(userID)

      val rec_artist_ids = rec_items.select("artist").as[Int].collect()
      artistData.filter($"id" isin (rec_artist_ids: _*)).show()
      model.userFactors.unpersist()
      model.itemFactors.unpersist()
    }

    def evaluate(
                  rawUserArtistData: Dataset[String],
                  rawArtistAlias: Dataset[String]): Unit = {
      val bArtistAlias = spark.sparkContext.broadcast(buildArtistAlias(rawArtistAlias))
      val allData = buildCounts(rawUserArtistData, bArtistAlias, spark)
      val Array(trainData, cvData) = allData.randomSplit(Array(0.9, 0.1))
      trainData.cache()
      cvData.cache()

    }




    def areaUnderCurve(
                        positiveData: DataFrame,
                        bAllArtistIDs: Broadcast[Array[Int]],
                        predictFunction: (DataFrame => DataFrame)): Double = {

      // What this actually computes is AUC, per user. The result is actually something
      // that might be called "mean AUC".

      // Take held-out data as the "positive".
      // Make predictions for each of them, including a numeric score
      val positivePredictions = predictFunction(positiveData.select("user", "artist")).
        withColumnRenamed("prediction", "positivePrediction")

      // BinaryClassificationMetrics.areaUnderROC is not used here since there are really lots of
      // small AUC problems, and it would be inefficient, when a direct computation is available.

      // Create a set of "negative" products for each user. These are randomly chosen
      // from among all of the other artists, excluding those that are "positive" for the user.
      val negativeData = positiveData.select("user", "artist").as[(Int, Int)].
        groupByKey { case (user, _) => user }.
        flatMapGroups { case (userID, userIDAndPosArtistIDs) =>
          val random = new Random()
          val posItemIDSet = userIDAndPosArtistIDs.map { case (_, artist) => artist }.toSet
          val negative = new ArrayBuffer[Int]()
          val allArtistIDs = bAllArtistIDs.value
          var i = 0
          // Make at most one pass over all artists to avoid an infinite loop.
          // Also stop when number of negative equals positive set size
          while (i < allArtistIDs.length && negative.size < posItemIDSet.size) {
            val artistID = allArtistIDs(random.nextInt(allArtistIDs.length))
            // Only add new distinct IDs
            if (!posItemIDSet.contains(artistID)) {
              negative += artistID
            }
            i += 1
          }
          // Return the set with user ID added back
          negative.map(artistID => (userID, artistID))
        }.toDF("user", "artist")

      // Make predictions on the rest:
      val negativePredictions = predictFunction(negativeData).
        withColumnRenamed("prediction", "negativePrediction")

      // Join positive predictions to negative predictions by user, only.
      // This will result in a row for every possible pairing of positive and negative
      // predictions within each user.
      val joinedPredictions = positivePredictions.join(negativePredictions, "user").
        select("user", "positivePrediction", "negativePrediction").cache()

      // Count the number of pairs per user
      val allCounts = joinedPredictions.
        groupBy("user").agg(count(lit("1")).as("total")).
        select("user", "total")
      // Count the number of correctly ordered pairs per user
      val correctCounts = joinedPredictions.
        filter($"positivePrediction" > $"negativePrediction").
        groupBy("user").agg(count("user").as("correct")).
        select("user", "correct")

      // Combine these, compute their ratio, and average over all users
      val meanAUC = allCounts.join(correctCounts, Seq("user"), "left_outer").
        select($"user", (coalesce($"correct", lit(0)) / $"total").as("auc")).
        agg(mean("auc")).
        as[Double].first()

      joinedPredictions.unpersist()

      meanAUC
    }

  }

}
