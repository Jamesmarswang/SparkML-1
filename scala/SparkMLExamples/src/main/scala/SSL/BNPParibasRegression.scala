import java.io.{File, BufferedWriter, FileWriter}

import breeze.linalg.support.LiteralRow
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import scala.util.Random

/**
  * Created by alexsisu on 10/03/16.
  */
object BNPParibasRegression extends App {
  val sparkConf = new SparkConf()
  sparkConf.setAppName("RandomForestRegressorKin8Data")
  sparkConf.setMaster("local")

  val sc = new SparkContext(sparkConf)
  val constantCategorical = -1111.11111d
  val constantNotFound = -99999999.9d

  val categoricalFeatures = List()
  val featuresToDrop = List(11, 19, 26, 43, 92, 113) // features that should be dropped for performance concerns
  val importantFeatures = List() // features that should be kept


  def findIdenticalColumns(info : String): Unit = {
    val allLines = info.split("\n")
    val names = allLines(0).split(",").drop(2)
    val linesToBeUsed = allLines.drop(1)//.filter(_ => Random.nextDouble() < 0.2d)
    val allItems = linesToBeUsed.map(line => (line + " ").split(","))
    val allColumns = (0 to (names.length + 1)).map {i =>
      println(i)
      allItems.map { line =>
        val value = line(i)
        if(value.matches("[a-zA-Z ]+[0-9]*") || value.size == 0) {
          if(value.length == 0) {
            constantNotFound
          }
          else if(value.matches("[A-Z]*")){
            constantCategorical
          }
          else {
            constantNotFound
          }
        }
        else
          value.toDouble
      }}.drop(2)
    println("non-raw data")
    val eachCol = allColumns.map(col => col.filter(_==constantNotFound).size / (col.size + 0.0d))
    names.zip(eachCol).foreach(println)
    val maxCol = allColumns.map(_.filter(value => value != constantNotFound)).map(_.max)
    val minCol = allColumns.map(_.filter(value => value != constantNotFound)).map(_.min)
    val normalizedColumns = allColumns.zip(0 to (allColumns.size - 1))
      .map{tuple => tuple._1.map { value =>
        if(value == constantCategorical)
          constantCategorical
        else if(value == constantNotFound)
          constantNotFound
        else
        (value - minCol(tuple._2)) / (maxCol(tuple._2) - minCol(tuple._2))
      }}
    println(maxCol)
    println(minCol)

    println("normalization finished")
    normalizedColumns.zip(names).foreach { tuple1 =>
      val col1 = tuple1._1
      val name1 = tuple1._2

      if(col1(0) != constantCategorical) {
        val validFields = col1.filter(_ != constantNotFound)
        val mean = validFields.foldLeft(0.0d)(_+_) / validFields.size
        val variance = validFields.map(x => Math.pow(x - mean, 2)).sum / validFields.size
        println(name1 + " " + variance)

        normalizedColumns.zip(names).foreach { tuple2 =>
          val col2 = tuple2._1
          val name2 = tuple2._2
          if (name1 != name2 && col2(0) != constantCategorical) {
            //println("Comparing " + name1 + " with " + name2)

            val colZip = col1.zip(col2).filter(t => t._1 != constantNotFound && t._2 != constantNotFound)
            val diff = colZip.map(t => t._1 - t._2)
            val mean = diff.foldLeft(0.0d)(_ + _) / col1.size
            val variance = diff.map(el => Math.pow(el - mean, 2)).foldLeft(0.0d)(_ + _) / diff.size

            if (variance < 0.0002) {
              println(name1 + " - " + name2 + ": " + variance)
              println("Total mse: " + variance)
              println("Common: " + colZip.size)
            }

          }

        }
      }
    }


  }

  def prepareData(info : String, typeOp : Int): RDD[LabeledPoint] = {
    var literalMap : Map[Int, Map[String, Double]] = Map()
    val allLines = info.split("\n")
    val recordsOnly = allLines.slice(1, allLines.length)
    val trainingDataInitial: RDD[LabeledPoint] = sc.parallelize(recordsOnly).map { case line =>
      val arr = (line + " ").split(",").zipWithIndex.map(tuple => {
        val(token, index) = tuple
        if(token.size == 0 || token.equals(" "))
          constantNotFound
        else if(token.matches("[a-zA-Z]*")) {
          literalMap.get(index) match {
            case Some(x) => {
              if (!x.contains(token)) {
                val smallMap: (String, Double) = (token -> (literalMap.get(index).get.size.toDouble + 1))
                val newInnerMap: Map[String, Double] = (Array(smallMap) ++ literalMap.get(index).get.toArray[(String, Double)]).toMap[String, Double]
                literalMap = literalMap ++ Map(index -> newInnerMap)
              }
            }
            case None => {
              val smallMap: Map[String, Double] = Map((token -> (literalMap.get(index).size.toDouble + 1)))
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
  def prepareData2(info : String, typeOp : Int): RDD[LabeledPoint] = {
    var literalMap : Map[Int, Map[String, Double]] = Map()
    val allLines = info.split("\n")
    val recordsOnly = allLines.slice(1, allLines.length)
    val trainingDataInitial = recordsOnly.map { case line =>
      val arr = (line + " ").split(",").zipWithIndex
        .filter(t => !featuresToDrop.contains(t._2 - 1 + typeOp))
        .map(_._1).zipWithIndex
        .map(tuple => {
          val (token, index) = tuple
          if (token.size == 0 || token.equals(" "))
            constantNotFound
          else if (token.matches("[a-zA-Z]*")) {
            literalMap.get(index) match {
              case Some(x) => {
                if (!x.contains(token)) {
                  val smallMap: (String, Double) = (token -> (literalMap.get(index).get.size.toDouble + 1))
                  val newInnerMap: Map[String, Double] = (Array(smallMap) ++ literalMap.get(index).get.toArray[(String, Double)]).toMap[String, Double]
                  literalMap = literalMap ++ Map(index -> newInnerMap)
                }
              }
              case None => {
                val smallMap: Map[String, Double] = Map((token -> (literalMap.get(index).size.toDouble + 1)))
                literalMap = Map(index -> smallMap) ++ literalMap
              }
            }

            literalMap.get(index).get.get(token).get
          }
          else {
            token.trim.toDouble
          }
        })

        arr
    }

    val trainingDataCol = (0 to (trainingDataInitial(0).length - 1)).map { i =>
      trainingDataInitial.map(_(i))
    }

    val initValidDataCol = trainingDataCol.map { col =>
      col.filter(_!=constantNotFound)
    }

    val initColMean = initValidDataCol.map { col => col.sum / col.size }
    val initVariance = initValidDataCol.zipWithIndex.map { t =>
      t._1.map { x =>
        Math.pow(x - initColMean(t._2), 2)
      }.sum / t._1.size
    }

    var newTrainingDataCol = trainingDataCol.zipWithIndex.map { t =>
      if(!literalMap.contains(t._2)) {
         t._1.map { x =>
           if(Math.abs(x - initColMean(t._2)) > 2 * initVariance(t._2)) {
             initColMean(t._2)
           }
           else {
             x
           }
         }
      }
      else {
        t._1
      }
    }
    if(typeOp == 1)
      newTrainingDataCol = trainingDataCol

    val colMax = newTrainingDataCol.map(x => x.max)
    val colMin = newTrainingDataCol.map(x => x.min)

    val trainingDataColNew = newTrainingDataCol.zipWithIndex.map { t=>
      val result = t._1.map { value =>
        if(literalMap.contains(t._2))
          value
        else if(value == constantNotFound)
          constantNotFound
        else
        if(value == constantCategorical)
          constantCategorical
        else if(value == constantNotFound)
          constantNotFound
        else
          (value - colMin(t._2)) / (colMax(t._2) - colMin(t._2))
      }
      val validCols = result.filter(_!=constantNotFound)
      val colMean = validCols.sum / validCols.size

      if(literalMap.contains(t._2))
        result.map(x => if(x == constantNotFound) 0 else x)
      else
        result.map(x => if(x == constantNotFound) colMean else x)
    }


    val intermed = (0 to (trainingDataColNew(0).length - 1)).map { i =>
      trainingDataColNew.map(_(i))
    }

    if(typeOp == 0) {
      val negatives = intermed.zip(trainingDataInitial).map { t =>
        (t._1, t._2(1))
      }.filter(t => t._2 == 0)

      val positives = intermed.zip(trainingDataInitial).map { t =>
        (t._1, t._2(1))
      }.filter(t => t._2 == 1)

      val necessary = intermed.size / 2 - negatives.size

      val trainingSet = (1 to necessary).map{ _ =>
        val index = Random.nextInt(negatives.size)
        negatives(index)
      } ++ negatives ++ positives

      sc.parallelize(trainingSet).map { data =>
        new LabeledPoint(data._2, Vectors.dense(data._1.toArray))
      }
    }
    else {
      sc.parallelize(intermed).map { data =>
        new LabeledPoint(1, Vectors.dense(data.toArray))
      }
    }
      /*
      *       if(typeOp == 0)
        new LabeledPoint(t._2(1),Vectors.dense(t._1.drop(2)))
      else
        new LabeledPoint(0,Vectors.dense(t._1.drop(1)))
      * */
  }

  def getCategories(info : String): Map[Int, Int] = {
    var literalMap : Map[Int, Map[String, Double]] = Map()
    val allLines = info.split("\n")
    val recordsOnly = allLines.slice(1, allLines.length)

    recordsOnly.map { case line =>
      val arr = (line + " ").split(",").zipWithIndex.map(tuple => {
        val(token, index) = tuple
        if(token.size == 0 || token.equals(" "))
          0.0d
        else if(token.matches("[a-zA-Z]*")) {
          literalMap.get(index) match {
            case Some(x) => {
              if (!x.contains(token)) {
                val smallMap: (String, Double) = (token -> (literalMap.get(index).get.size.toDouble + 1))
                val newInnerMap: Map[String, Double] = (Array(smallMap) ++ literalMap.get(index).get.toArray[(String, Double)]).toMap[String, Double]
                literalMap = literalMap ++ Map(index -> newInnerMap)
              }
            }
            case None => {
              val smallMap: Map[String, Double] = Map((token -> (literalMap.get(index).size.toDouble + 1)))
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
    literalMap.map(t => (t._1 - 2, t._2.size + 1)).filter(t => t._2 < 16 && t._1 >= 0)
  }

  val trainText = scala.io.Source.fromFile("C:\\Users\\Bogdan\\IdeaProjects\\SparkML\\scala\\SparkMLExamples\\resources\\train_bnpparibas.csv").mkString
  //val trainText = scala.io.Source.fromFile("C:\\Users\\BDN\\IdeaProjects\\SparkML\\scala\\SparkMLExamples\\resources\\train_bnpparibas.csv\\train.csv").mkString
  //val testText = scala.io.Source.fromFile("C:\\Users\\BDN\\IdeaProjects\\SparkML\\scala\\SparkMLExamples\\resources\\test_bnpparibas.csv\\test.csv").mkString

  val data: RDD[LabeledPoint] = prepareData2(trainText, 0)
  //findIdenticalColumns(trainText)
  println("Finished reading data")

  val splits = data.randomSplit(Array(0.7, 0.3))
  val (trainingData, validationData) = (splits(0), splits(1))

  val categoricalFeaturesInfo = getCategories(trainText)
  val numTrees = 20 // Use more in practice.
  val featureSubsetStrategy = "auto" // Let the algorithm choose.
  val impurity = "variance"
  val maxDepth = 9
  val maxBins = 16

  categoricalFeaturesInfo.foreach(println)
  val model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo,
    numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

  //val testData: RDD[LabeledPoint] = prepareData(testText, 1)
  //val gt = testData.map(labeledPoint => labeledPoint.label)

  val result = model.predict(validationData.map(l => l.features))
  val gt = validationData.map(labeledPoint => labeledPoint.label)
  val res = gt.zip(result).collect

  //res.take(10).foreach(println)
  val logloss =res.map {tuple => if(tuple._2 == 0.0d)
    tuple._1 * Math.log(tuple._2 + 0.0001)
  else if(tuple._2 == 1)
    tuple._1 * Math.log(tuple._2 - 0.0001)
  else
    tuple._1 * Math.log(tuple._2)
  }.foldLeft(0.0d){ (x: Double, y: Double) => x+y }

  val resultingLogLoss = - logloss/res.length
  println(resultingLogLoss)

/*
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
*/
  Thread.sleep(100000)
}
