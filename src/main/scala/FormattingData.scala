import org.apache.spark.ml.feature.{Bucketizer, MinMaxScaler, QuantileDiscretizer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, udf}

object FormattingData extends App {
  val spark = SparkSession.builder().appName("test").master("local").getOrCreate()
  println(s"Session started on Spark version ${spark.version}")

  val sales = spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load("./src/resources/retail-data/by-day/*.csv") //we took multiple csv files and loaded them together into a single DF
    .coalesce(5)
    .where("Description IS NOT NULL")
//  val fakeIntDF = spark.read.parquet("./src/resources/simple-ml-integers")
  val fakeIntDF = spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load("./src/resources/my-ints.csv") //we took multiple csv files and loaded them together into a single DF
  var simpleDF = spark.read.json("./src/resources/simple-ml")
  val scaleDF = spark.read.parquet("./src/resources/simple-ml-scaling")
  fakeIntDF.printSchema()
  fakeIntDF.show(truncate = false)

  sales.cache()
  sales.show(5, truncate= false)

  //The Tokenizer is an example of a transformer. It tokenizes a string, splitting on a given
  //character, and has nothing to learn from our data; it simply applies a function
  import org.apache.spark.ml.feature.Tokenizer
  val tkn = new Tokenizer().setInputCol("Description")
  tkn.transform(sales.select("Description")).show(5, false)

  //StandardScaler, which scales your input column
  //according to the range of values in that column to have a zero mean and a variance of 1 in each
  //dimension. For that reason it must first perform a pass over the data to create the transformer.
  // in Scala
  scaleDF.printSchema()
  scaleDF.show(5, truncate = false)
  import org.apache.spark.ml.feature.StandardScaler
  val ss = new StandardScaler()
    .setInputCol("features")
    .setOutputCol("scaled_features")
  ss.fit(scaleDF).transform(scaleDF).show(false)



  import org.apache.spark.ml.feature.RFormula
  val supervised = new RFormula()
    .setFormula("lab ~ . + color:value1 + color:value2")
  supervised.fit(simpleDF).transform(simpleDF).show(truncate = false)

  val intFormula = new RFormula()
    .setFormula("int1 ~ . + int2 + int3 + int2:int3") //so : means multiply
  val intFeatureLab = intFormula.fit(fakeIntDF).transform(fakeIntDF)

    val iScaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("features_scaled")
  iScaler.fit(intFeatureLab).transform(intFeatureLab).show(truncate = false)

  // in Scala
  //Any SELECT statement you can use in SQL is a valid
  //transformation. The only thing you need to change is that instead of using the table name, you
  //should just use the keyword THIS.
  import org.apache.spark.ml.feature.SQLTransformer
  val basicTransformation = new SQLTransformer()
    .setStatement("""
    SELECT sum(Quantity), count(*), CustomerID
    FROM __THIS__
    GROUP BY CustomerID
    """)
  basicTransformation.transform(sales).show(5, truncate=false)

  //so you can data munge/transform as much as you want in SQL
  //https://spark.apache.org/docs/latest/api/sql/index.html
  val intTransformation = new SQLTransformer()
    .setStatement("""
    SELECT *, int1*int2 as int1_int2, array(int1) , array(int2,int3+10) as features
    FROM __THIS__
    """)
  //so __THIS__ refers to whichever DataFrame you are transforming
  //below it would be fakeIntDF
//  intTransformation.transform(fakeIntDF).show(truncate=false)

  val transformedDF = intTransformation.transform(fakeIntDF)
  transformedDF.printSchema()
  transformedDF.show(truncate = false)

  val va = new VectorAssembler()
    .setInputCols(Array("int1", "int2", "int3"))
    .setOutputCol("features") //in case we do not want some random column name
  val intWithFeatures = va.transform(fakeIntDF)
  intWithFeatures.show()

  val contDF = spark.range(20).selectExpr("cast(id as double)")
  contDF.show()

  //When specifying your
  //bucket points, the values you pass into splits must satisfy three requirements:
  //The minimum value in your splits array must be less than the minimum value in your
  //DataFrame.
  //The maximum value in your splits array must be greater than the maximum value in
  //your DataFrame.
  //You need to specify at a minimum three values in the splits array, which creates two
  //buckets.
  //WARNING
  //The Bucketizer can be confusing because we specify bucket borders via the splits method, but these

  val bucketBorders = Array(-1.0, 5.0, 10.0, 250.0, 600.0)
  val bucketer = new Bucketizer().setSplits(bucketBorders).setInputCol("id")
  bucketer.transform(contDF).show()

  val quantBucketer = new QuantileDiscretizer()
    .setNumBuckets(5)
    .setInputCol("id")
  val fittedBucketer = quantBucketer.fit(contDF)
  fittedBucketer.transform(contDF).show()

  //The StandardScaler standardizes a set of features to have zero mean and a standard deviation
  //of 1. The flag withStd will scale the data to unit standard deviation while the flag withMean
  //(false by default) will center the data prior to scaling it.
val sScaler = new StandardScaler()
    .setInputCol("features")
    .setWithMean(true)
    .setWithStd(true)
    .setOutputCol("mean_scaled")


  scaleDF.show()
  sScaler
    .fit(scaleDF)
    .transform(scaleDF)
    .show(truncate = false) //fit was necessary because we had to calculate mean first

  //TODO exercise scale our transformedDF with StandardScaler


  //The MinMaxScaler will scale the values in a vector (component wise) to the proportional values
  //on a scale from a given min value to a max value. If you specify the minimum value to be 0 and
  //the maximum value to be 1, then all the values will fall in between 0 and 1:
  val minMax = new MinMaxScaler()
    .setMin(5)
    .setMax(10)
    .setInputCol("features")
    .setOutputCol("scaled_features")
  val fittedMinMax = minMax.fit(scaleDF)
  fittedMinMax.transform(scaleDF).show(truncate=false)

  println("ScaleDF Schema")
  scaleDF.printSchema()


//  The max absolute scaler (MaxAbsScaler) scales the data by dividing each value by the maximum
//  absolute value in this feature. All values therefore end up between −1 and 1. This transformer
//    does not shift or center the data at all in the process:
  import org.apache.spark.ml.feature.MaxAbsScaler
  val maScaler = new MaxAbsScaler().setInputCol("features")
  val fittedAbsScaler = maScaler.fit(scaleDF)
  fittedAbsScaler.transform(scaleDF).show()

//  The ElementwiseProduct allows us to scale each value in a vector by an arbitrary value. For
//  example, given the vector below and the row “1, 0.1, -1” the output will be “10, 1.5, -20.”
//  Naturally the dimensions of the scaling vector must match the dimensions of the vector inside
//  the relevant column:
  import org.apache.spark.ml.feature.ElementwiseProduct
  import org.apache.spark.ml.linalg.Vectors
  val scaleUpVec = Vectors.dense(10.0, 15.0, 20.0)
  val scalingUp = new ElementwiseProduct()
    .setScalingVec(scaleUpVec)
    .setInputCol("features")
  scalingUp.transform(scaleDF).show()

  //so we need to convert array of some ints into a Vector of Doubles
  //so we will create a user defined function
  val convertUDF = udf((array : Seq[Int]) => {
    Vectors.dense(array.toArray.map(_.toDouble))
  })

  val afWithVector = transformedDF.select("*").withColumn("features", convertUDF(col("features")))
  afWithVector.printSchema()
  afWithVector.show(truncate=false)

  val zeroScaler = new MinMaxScaler()
    .setMin(0)
    .setMax(1)
    .setInputCol("features")
    .setOutputCol("scaled_features")

  //lets try different scalers on our df with Vector
  zeroScaler.fit(afWithVector).transform(afWithVector).show(truncate=false)
  minMax.fit(afWithVector).transform(afWithVector).show(truncate=false)

  sScaler
    .fit(afWithVector)
    .transform(afWithVector)
    .show(truncate = false)

  //TODO check scaler options
}