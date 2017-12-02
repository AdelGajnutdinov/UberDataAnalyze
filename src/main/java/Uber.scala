import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.sql.types.TimestampType
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.clustering.KMeans

object Uber {

  case class Uber(dt: String, lat: Double, lon: Double, base: String) extends Serializable

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:\\winutil\\")

    //Initialize SparkSession
    val sparkSession = SparkSession
      .builder()
      .appName("spark-uber-analysis")
      .master("local[*]")
      .getOrCreate()

    import sparkSession.implicits._
    val inputFile = "C:/dataForScalaProjects/uber.csv"

    val schema = StructType(Array(
      StructField("dt", TimestampType, true),
      StructField("lat", DoubleType, true),
      StructField("lon", DoubleType, true),
      StructField("base", StringType, true)
    ))

    //Load Uber Data to DF
    val uberData = sparkSession.read
      .option("header", "true")
      .option("inferSchema", "false")
      .schema(schema)
      .csv(inputFile)
      .as[Uber]

    //uberData.createOrReplaceTempView("uber")

    //uberData.show(50)
    //uberData.printSchema()
    //getListWithHighestCountOfTrips(sparkSession).show
    // Get Feature Vectors

    val featureCols = Array("lat", "lon")
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val uberFeatures = assembler.transform(uberData)

    //Split data into training and testing data
    val Array(trainingData, testData) = uberFeatures.randomSplit(Array(0.7, 0.3), 5043)

    //Traing KMeans model
    val kmeans = new KMeans()
      .setK(20)
      .setFeaturesCol("features")
      .setMaxIter(20)

    val model = kmeans.fit(trainingData)

    println("Final Centers: ")
    model.clusterCenters.foreach(println)

    //Get Predictions
    val predictions = model.transform(testData)
    predictions.show
    predictions.createOrReplaceTempView("uberWithPredictions")

    //Which hours of the day and which cluster had the highest number of pickups?
    predictions.select(hour($"dt").alias("hour"), $"prediction")
      .groupBy("hour", "prediction").agg(count("prediction")
      .alias("count"))
      .orderBy(desc("count"))
//      .show

    val res = sparkSession.sql("select dt, lat, lon, base, prediction as cid FROM uberWithPredictions where prediction = 1")
    res.coalesce(1).write.format("json").save("C:/dataForScalaProjects/uber.json")
  }

  //Get the list of companies with have the highest count of trips
  def getListWithHighestCountOfTrips(sparkSession : SparkSession): DataFrame = {
    sparkSession.sql(
      " SELECT base, count(*)" +
        " FROM uber" +
        " GROUP BY base ORDER BY count(*) DESC")
  }
}
