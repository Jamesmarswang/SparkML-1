import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint

import scala.collection.immutable.Map
import scala.util.control.NonFatal


class BNPDataEntry(line: String) {
  var features: Map[String, String] = {
    val record = line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)", -1)
    val Array(crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat, medv) = record

    Map("crim" -> crim, "zn" -> zn, "indus" -> indus,
      "chas" -> chas, "nox" -> nox, "rm" -> rm, "age" -> age, "dis" -> dis,
      "rad" -> rad, "tax" -> tax, "ptratio" -> ptratio, "b" -> b, "lstat" -> lstat, "medv" -> medv)
  }

  def parseDouble(s: String, default: Double = 0): Double = try {
    s.toDouble
  } catch {
    case NonFatal(_) => default
  }

  def toVector(): Vector = {
    return Vectors.dense(
      parseDouble(features("crim")),
      parseDouble(features("zn")),
      parseDouble(features("indus")),
      parseDouble(features("chas")),
      parseDouble(features("nox")),
      parseDouble(features("rm")),
      parseDouble(features("age")),
      parseDouble(features("dis")),
      parseDouble(features("rad")),
      parseDouble(features("tax")),
      parseDouble(features("ptratio")),
      parseDouble(features("b")),
      parseDouble(features("lstat"))
    )
  }

  def toLabeledPoint(): LabeledPoint = {
    return LabeledPoint(parseDouble(features("medv")), toVector())
  }
}
