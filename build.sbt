name := "Spark"

version := "0.1"

scalaVersion := "2.12.12"


// https://mvnrepository.com/artifact/org.apache.spark/spark-core
libraryDependencies += "org.apache.spark" %% "spark-core" % "3.0.1"
// https://mvnrepository.com/artifact/org.apache.spark/spark-sql
libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.0.1"

// https://mvnrepository.com/artifact/org.apache.spark/spark-mllib
//libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.0.1" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.0.1"


libraryDependencies += "com.github.mrpowers" %% "spark-daria" % "0.38.2"

libraryDependencies += "org.xerial" % "sqlite-jdbc" % "3.32.3.2"

libraryDependencies +=  "org.scalaj" %% "scalaj-http" % "2.4.2"
