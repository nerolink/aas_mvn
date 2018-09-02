package chapter5

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.util.Random

object RunKMeans_self {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.set("spark.default.parallelism", "4")
    conf.set("spark.master", "local")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    import spark.implicits._
    val dataWithoutHead=spark.read.option("inferSchema", true).option("header", false).csv("./kddcup.data.corrected")
    val data=dataWithoutHead.toDF(  "duration", "protocol_type", "service", "flag",
      "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
      "hot", "num_failed_logins", "logged_in", "num_compromised",
      "root_shell", "su_attempted", "num_root", "num_file_creations",
      "num_shells", "num_access_files", "num_outbound_cmds",
      "is_host_login", "is_guest_login", "count", "srv_count",
      "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
      "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
      "dst_host_count", "dst_host_srv_count",
      "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
      "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
      "dst_host_serror_rate", "dst_host_srv_serror_rate",
      "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
      "label")
    data.cache()


  }
}

class RunKMeans(private val spark:SparkSession){
  import spark.implicits._

  def clusteringTask0(data:DataFrame): Unit = {
    data.select("label").groupBy("label").count().orderBy($"count".desc).show()
    val numOnly= data.drop("protocol_type", "service", "flag").cache()
    val assembler=new VectorAssembler().setInputCols(numOnly.columns.filter(_ != "label")).setOutputCol("featuresVector")
    val k_means= new KMeans().setSeed(Random.nextLong()).setFeaturesCol("featuresVector").setPredictionCol("cluster")
    val pipeline=new Pipeline().setStages(Array(assembler, k_means))
    val pipelineModel=pipeline.fit(numOnly)
    val k_meansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
    k_meansModel.clusterCenters.foreach(println)

    val withCluster= pipelineModel.transform(numOnly)
    withCluster.select("cluster","label").groupBy("cluster","label").count().orderBy($"cluster",$"label".desc)
  }

}