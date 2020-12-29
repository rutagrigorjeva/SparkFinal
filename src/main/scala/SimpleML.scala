import org.apache.spark.ml.feature.RFormula
import org.apache.spark.sql.SparkSession

object SimpleML extends App {
  val session = SparkSession.builder().appName("test").master("local").getOrCreate()
  println(s"Session started on Spark version ${session.version}")

  // in Scala
  var df = session.read.json("./src/resources/simple-ml")
  df.orderBy("value2").show()

  val supervised = new RFormula()
      .setFormula("lab ~ . + color:value1 + color:value2")

  //apply formula
  val fittedRF = supervised.fit(df)

  val preparedDF = fittedRF.transform(df)
  preparedDF.show(truncate = false)

  //in supervised learning we want to train on one data set and test on completely separate data set
  val Array(train, test) = preparedDF.randomSplit(Array(0.7, 0.3)) //so 70% for training and 30% for testing

  train.describe().show()

  test.describe().show()

  //how to save on Parquet
//  preparedDF
//    .write
//    .format("parquet")
//    .mode("overwrite")
//    .save("./src/resources/simple-ml.parquet")
//
//  val newPath = "./src/resources/simple-ml.parquet"
//  val newDF = session.read
//    .format("parquet")
////    .option("inferSchema", "true") // for parquet all the data types are encoded
////    .option("header", true)
//    .load(newPath)
//  newDF.printSchema()
//  newDF.show(truncate = false)



  //actually  model is fitted here, so to say rubber meets the road here
  // in Scala
//  val fittedLR = lr.fit(train)
//
//  //now we should be able to do predictions and test our accuracy
//  //so here transform would be predicting
//  fittedLR.transform(train).select("label", "prediction").show()
//
//  //how about using it on test set?
//  fittedLR.transform(test).select("label", "prediction").show()


  //creating a Pipeline which can combine the above in a more standard way
  // in Scala
  import org.apache.spark.ml.classification.LogisticRegression
  val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features")
  val rForm = new RFormula()
  import org.apache.spark.ml.Pipeline
  val stages = Array(rForm, lr)
  val pipeline = new Pipeline().setStages(stages)

  println("Going to add Parameters")

  import org.apache.spark.ml.tuning.ParamGridBuilder
  val params = new ParamGridBuilder() //FIXME rForm formula
    .addGrid(rForm.formula, Array(
      "lab ~ . + color:value1",
      "lab ~ . + color:value1 + color:value2"))
    .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
    .addGrid(lr.regParam, Array(0.1, 2.0))
    .build()

  // in Scala
  import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
  val evaluator = new BinaryClassificationEvaluator()
    .setMetricName("areaUnderROC")
    .setRawPredictionCol("prediction")
    .setLabelCol("label")

  import org.apache.spark.ml.tuning.TrainValidationSplit
  val tvs = new TrainValidationSplit()
    .setTrainRatio(0.75) // also the default.
    .setEstimatorParamMaps(params)
    .setEstimator(pipeline)
    .setEvaluator(evaluator)


  //so the whole work of testing 12 different hyperparameter combinations
  //and evaluating the accuracy is done in a single short line

  println("Ready to Fit!")
//  val tvsFitted = tvs.fit(train) //FIXME column features already exist
  val tvsFitted = tvs.fit(df)

  //for kick we can see how well it works on the training set
  //it can be as high as 100% here but that doesnt mean anything
//  println("Training acc:", evaluator.evaluate(tvsFitted.transform(train)))

  //Crucially we can see how well it works on our test set
  println("TRAIN ACC:", evaluator.evaluate(tvsFitted.transform(df)))// 0.9166666666666667
  //FIXME we need to test on TRAIN SET not the whole DF!

  import org.apache.spark.ml.PipelineModel
  import org.apache.spark.ml.classification.LogisticRegressionModel
  val trainedPipeline = tvsFitted.bestModel.asInstanceOf[PipelineModel]
  val TrainedLR = trainedPipeline.stages(1).asInstanceOf[LogisticRegressionModel]
  val summaryLR = TrainedLR.summary
  summaryLR.objectiveHistory.foreach(println) // 0.6751425885789243, 0.5543659647777687, 0.473776..

  tvsFitted.write.overwrite().save("./src/resources/models")

  import org.apache.spark.ml.tuning.TrainValidationSplitModel
  val model = TrainValidationSplitModel.load("./src/resources/models")
  val newDF = model.transform(df)
  newDF.printSchema()
  newDF.show(10, truncate = false)
}