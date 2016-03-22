import java.io.{File, BufferedWriter, FileWriter}

import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by alexsisu on 10/03/16.
  */
object BNPParibasRegression extends App {
  val sparkConf = new SparkConf()
  sparkConf.setAppName("RandomForestRegressorKin8Data")
  sparkConf.setMaster("local")

  val sc = new SparkContext(sparkConf)

  def prepareData(info : String, typeOp : Int): RDD[LabeledPoint] = {
    var literalMap : Map[Int, Map[String, Double]] = Map()
    val allLines = info.split("\n")
    val recordsOnly = allLines.slice(1, allLines.length)
    val trainingDataInitial: RDD[LabeledPoint] = sc.parallelize(recordsOnly).map { case line =>
      val arr = (line + " ").split(",").zipWithIndex.map(tuple => {
        val(token, index) = tuple
        if(token.size == 0 || token.equals(" "))
          -1.0d
        else if(token.matches("[a-zA-Z]*")) {
          literalMap.get(index) match {
            case Some(x) => {
              if (!x.contains(token)) {
                val smallMap: (String, Double) = (token -> literalMap.get(index).get.size.toDouble)
                val newInnerMap: Map[String, Double] = (Array(smallMap) ++ literalMap.get(index).get.toArray[(String, Double)]).toMap[String, Double]
                literalMap = literalMap ++ Map(index -> newInnerMap)
              }
            }
            case None => {
              val smallMap: Map[String, Double] = Map((token -> literalMap.get(index).size.toDouble))
              literalMap = Map(index -> smallMap) ++ literalMap
            }
          }

          literalMap.get(index).get.get(token).get
        }
        else {
          token.trim.toDouble
        }
      })
      //println(arr.size)
      if(typeOp == 0)
        new LabeledPoint(arr(1),Vectors.dense(arr.slice(2,arr.size - 1)))
      else
        new LabeledPoint(arr(0),Vectors.dense(arr.slice(1,arr.size - 1)))
    }
    trainingDataInitial
  }
  def getCategories(info : String): Map[Int, Int] = {
    var literalMap : Map[Int, Map[String, Double]] = Map()
    val allLines = info.split("\n")
    val recordsOnly = allLines.slice(1, allLines.length)

    recordsOnly.map { case line =>
      val arr = (line + " ").split(",").zipWithIndex.map(tuple => {
        val(token, index) = tuple
        if(token.size == 0 || token.equals(" "))
          -1.0d
        else if(token.matches("[a-zA-Z]*")) {
          literalMap.get(index) match {
            case Some(x) => {
              if (!x.contains(token)) {
                val smallMap: (String, Double) = (token -> literalMap.get(index).get.size.toDouble)
                val newInnerMap: Map[String, Double] = (Array(smallMap) ++ literalMap.get(index).get.toArray[(String, Double)]).toMap[String, Double]
                literalMap = literalMap ++ Map(index -> newInnerMap)
              }
            }
            case None => {
              val smallMap: Map[String, Double] = Map((token -> literalMap.get(index).size.toDouble))
              literalMap = Map(index -> smallMap) ++ literalMap
            }
          }

          literalMap.get(index).get.get(token).get
        }
        else {
          token.trim.toDouble
        }
      })
    }
    literalMap.map(t => (t._1 - 2, t._2.size)).filter(t => t._2 < 128 && t._1 > 1)
  }

  val trainText = scala.io.Source.fromFile("resources/train_bnpparibas.csv").mkString
  val testText = scala.io.Source.fromFile("resources/test_bnpparibas.csv").mkString

  val trainData: RDD[LabeledPoint] = prepareData(trainText, 0)
  println("Finished reading data")

  println("\n\n\n\nLol Lol Lol\n\nn\n")
  val categoricalFeaturesInfo = getCategories(trainText)
  val numTrees = 30 // Use more in practice.
  val featureSubsetStrategy = "auto" // Let the algorithm choose.
  val impurity = "variance"
  val maxDepth = 9
  val maxBins = 128

  categoricalFeaturesInfo.foreach(println)
  val model = RandomForest.trainRegressor(trainData, categoricalFeaturesInfo,
    numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

  val testData: RDD[LabeledPoint] = prepareData(testText, 1)
  val gt = testData.map(labeledPoint => labeledPoint.label)
  val result = model.predict(testData.map(l => l.features))

  val res = gt.zip(result).collect
  val bw = new BufferedWriter(new FileWriter(new File("result.csv")))

  bw.write("ID,PredictedProb\n")
  res.foreach(x => bw.write(x._1.toInt + "," + x._2 + "\n"))
  bw.close()

  // Instantiate metrics object
  val metrics = new RegressionMetrics(gt.zip(result))

  // Squared error
  println(s"MSE = ${metrics.meanSquaredError}")
  println(s"RMSE = ${metrics.rootMeanSquaredError}")

  // R-squared
  println(s"R-squared = ${metrics.r2}")

  // Mean absolute error
  println(s"MAE = ${metrics.meanAbsoluteError}")

  // Explained variance
  println(s"Explained variance = ${metrics.explainedVariance}")

  Thread.sleep(100000)
}
