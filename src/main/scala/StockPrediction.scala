import java.io.{BufferedWriter, File, FileWriter}
import java.net.URLEncoder
import java.time.LocalDate
import java.time.format.DateTimeFormatter

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, expr, udf}
import org.apache.spark.sql.types.IntegerType
import scalaj.http.Http

object StockPrediction extends App {
  val companys = Map("AAPL" -> "Apple, Inc.",
    "SBUX" -> "Starbucks, Inc.",
    "MSFT" -> "Microsoft, Inc.",
    "CSCO" -> "Cisco Systems, Inc.",
    "QCOM" -> "QUALCOMM Incorporated",
    "FB" -> "Facebook, Inc.",
    "AMZN" -> "Amazon.com, Inc.",
    "TSLA" -> "Tesla, Inc.",
    "AMD" -> "Advanced Micro Devices, Inc.",
    "ZNGA" -> "Zynga Inc.")

  downloadData(companys)

  val session = SparkSession.builder().appName("test").master("local").getOrCreate()
  println(s"Session started on Spark version ${session.version}")
  val dfs = createDataFrames(companys)

  private def createDataFrames(companys: Map[String, String]): Map[String, DataFrame] = {
    companys.map(entry => {
      val companyShortcut = entry._1
      var df = session.read.format("csv")
        .option("header", "true")
        .load(s"./$companyShortcut.csv").toDF();
      df = df.columns.foldLeft(df)((curr, n) => curr.withColumnRenamed(n, n.trim))
      val removeDollarAndCastToDouble = udf((price: String) => {
        price.substring(2).toDouble
      })
      df = df.withColumn("Open", removeDollarAndCastToDouble(col("Open")))
      df = df.withColumn("High", removeDollarAndCastToDouble(col("High")))
      df = df.withColumn("Low", removeDollarAndCastToDouble(col("Low")))
      df = df.withColumn("Close", removeDollarAndCastToDouble(col("Close/Last")))
      df = df.withColumn("Volume", df("Volume").cast(IntegerType).as("Volume"))
      df = df.withColumn("Frequency", expr("Close * Volume"))
      df = df.withColumn("average_return", expr("(Close - Open)/Close*100"))
      df.printSchema()
      df.show(truncate = false)
      val avRet = df.select("Date", "average_return")
      avRet.show(truncate = false)
      avRet.write.mode("overwrite").parquet(s"$companyShortcut.parquet")
      entry._1 -> df
    }).toSeq.toMap
  }
  val frequnecy = dfs.map { case (name, df) => name -> df.agg("Frequency" -> "avg").collect()(0)(0).toString.toDouble }.toMap
  println(frequnecy.maxBy { case (name, frequency) => frequency })

  def downloadData(companys: Map[String, String]): Unit = {
    val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd")
    val currentDate = LocalDate.now();
    val dateEnd = formatter.format(currentDate);
    val dateStart = formatter.format(currentDate.minusYears(1));
    val template = "https://www.nasdaq.com/api/v1/historical/{companyShortcut}/stocks/{dateStart}/{dateEnd}"
    for ((companyShortcut, companyFullName) <- companys) {
      val url = template.replace("{companyShortcut}", URLEncoder.encode(companyShortcut, "UTF-8")).replace("{dateStart}", dateStart).replace("{dateEnd}", dateEnd)
      val result = Http(url).asString
      val body = result.body
      writeFile(companyShortcut + ".csv", body)
    }
  }

  def writeFile(filename: String, s: String): Unit = {
    val file = new File(filename)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(s)
    bw.close()
  }
}
