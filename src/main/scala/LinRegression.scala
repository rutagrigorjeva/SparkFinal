import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{DoubleType, FloatType, IntegerType, StructField, StructType}

object LinRegression extends App {
  val spark = SparkSession.builder().appName("test").master("local").getOrCreate()
  println(s"Session started on Spark version ${spark.version}")
  val df = spark.read.load("./src/resources/regression")
  df.printSchema()
  df.show(10,false)
  df.selectExpr("label").distinct().show()

  import org.apache.spark.ml.regression.LinearRegression
  val lr = new LinearRegression() //defaults are fine
//  //    .setMaxIter(10)
//  //    .setRegParam(0.3)
//  //    .setElasticNetParam(0.8) //one of these has a bug/feature that makes all values same
//
//  println(lr.explainParams())
//  val lrModel = lr.fit(df)
//
//  val summary = lrModel.summary
//
//  import spark.implicits._ //some implicit magic toDF
//  summary.residuals.show()
//  println(summary.objectiveHistory.toSeq.toDF.show())
//  println(summary.rootMeanSquaredError)
//  println(summary.r2)
//
//  val grades = spark.read.format("csv")
//    .option("header", "true")
//    .option("inferSchema", "true")
//    .load("./src/resources/simple_lin_regr.csv") //we took multiple csv files and loaded them together into a single DF
//
//  grades.printSchema()
//  grades.show(false)
//
//  val convertUDF = udf((element: Int) => {
//    Vectors.dense(element)
//  })
//
//
//  val gdf = grades
//    .withColumnRenamed("SAT", "features")
//    .withColumnRenamed("GPA", "label")
//    .select("*").withColumn("features", convertUDF(col("features")))
//
//  gdf.printSchema()
//  gdf.show(false)
//
//
//  val gradeModel = lr.fit(gdf)
//  val gradeSummary = gradeModel.summary
//  gradeSummary.residuals.show()
//
//  gdf.show(5, false)
//  gradeModel.transform(gdf).show(5,false)
//
//  gradeModel.transform(gdf.selectExpr("features")).show(15, false)
//
//  val inputs = Seq(Row(Vectors.dense(1200)),
//    Row(Vectors.dense(1500)),
//    Row(Vectors.dense(1800)),
//    Row(Vectors.dense(2100)),
//  )
//
//  val someSchema = List(
//    StructField("features", VectorType, true)
//  )
//  val inDF = spark.createDataFrame(
//    spark.sparkContext.parallelize(inputs),
//    StructType(someSchema)
//  )
//
//  inDF.printSchema()
//  inDF.show(false)
//
//  gradeModel.transform(inDF).show(false)
//
//  //TODO get the coefficients
//  // Here are the coefficient and intercept
//  //  val weights: org.apache.spark.mllib.linalg.Vector = gradeModel
//  val weights = gradeModel.coefficients
//  val intercept = gradeModel.intercept
//  val weightsData: Array[Double] = weights.asInstanceOf[DenseVector].values
//
//  println(s"Intercept is $intercept")
//  weightsData.foreach(println)
//  //  weightsData.foreach(coef: Double => println(s"Coef is $coef"))
//
//  import org.apache.spark.ml.regression.GeneralizedLinearRegression
//  val glr = new GeneralizedLinearRegression()
//    .setFamily("gaussian")
//    .setLink("identity")
//    .setMaxIter(10)
//    .setRegParam(0.3)
//    .setLinkPredictionCol("linkOut")
//  println(glr.explainParams())
//  val glrModel = glr.fit(gdf)
//  println(s"Intercept is ${glrModel.intercept}")
//  glrModel.coefficients.asInstanceOf[DenseVector].values.foreach(println)
//
//
//  //The residuals
//  //The difference between the label and the predicted value.
//  lrModel.summary.residuals.show(5)
//  glrModel.summary.residuals.show(5)
//  println("Default LR Model")
//  println("rootMeanSquaredError", lrModel.summary.rootMeanSquaredError)
//  println("r2", lrModel.summary.r2)
//  println("Generalized LR Model with some hyperparameters set")
//  println("Deviance", glrModel.summary.deviance)
//  //  println(glrModel.summary.r2)

  //R squared
  //The coefficient of determination; a measure of fit. so bas

  val numData = Seq(Row(2.0, 24.0), Row(2.5, 26.0), Row(3.0,28.1),Row(3.5,29.8))
  //  what will be the value at 5
  val numSchema = List(
    StructField("age", DoubleType, true),
    StructField("weight", DoubleType, true)
  )
  val someDF = spark.createDataFrame(
    spark.sparkContext.parallelize(numData),
    StructType(numSchema)
  )
  someDF.printSchema()
  someDF.show(false)

  val convertNumUDF = udf((element: Double) => {
    Vectors.dense(element)
  })
  val numDF = someDF
    .withColumn("features", convertNumUDF(col("age")))
    .withColumnRenamed("weight","label")
  numDF.printSchema()
  numDF.show()

  val weightModel = lr.fit(numDF) //all the work is done here once you have your features and label ready

  println(weightModel.intercept)
  println(weightModel.coefficients)
  println(weightModel.summary.residuals)


  val inputs = Seq(
    Row(Vectors.dense(2)),
    Row(Vectors.dense(4)),
    Row(Vectors.dense(5)),
    Row(Vectors.dense(24)),
    Row(Vectors.dense(29)),
    Row(Vectors.dense(50)),
  )

  val someSchema = List(
    StructField("features", VectorType, true)
  )
  val inDF = spark.createDataFrame(
    spark.sparkContext.parallelize(inputs),
    StructType(someSchema)
  )
  weightModel.transform(inDF).show(false)
}