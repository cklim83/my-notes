---
layout: page
title: Spark
nav_order: 4
permalink: /spark/
---

**J**ust **E**nough PySpark DataFrame
=====================================
*Last Edited: 10 June 2019*

## Table of Contents
- [Overview of PySpark Dataframe](#overview-of-pyspark-dataframe)
- [Libraries and Spark Instantiation](#libaries-and-spark-instantiation)
- [Creating DataFrames](#creating-dataframes)
- [Inspecting DataFrames](#inspecting-dataframes)
- [Missing & Replacing Values](#missing-&-replacing-values)
- [Filter Rows](#filter-rows)
- [Data Selection](#data-selection)
- [Add, Rename & Drop Columns](#add,-rename-&-drop-columns)
- [Remove Duplicates](#remove-duplicates)
- [GroupBy](#groupby)
- [Sort / OrderBy](#sort-/-orderby)
- [Repartitioning](#repartitioning)
- [Register DataFrame as View for SQL Type Queries](#register-dataframe-as-view-for-sql-type-queries)
- [Convert to Other Data Structure](#convert-to-other-data-structure)
- [Write to Files](#write-to-files)
- [Stop SparkSession](#stop-sparksession)



#### Overview of PySpark DataFrame
Beginning version 2.0, Spark has moved to a DataFrame API, a higher level abstraction built on top of RDD, the underlying data structure in Spark. Although a Spark DataFrame is very similar to Dataframe in R or Pandas, there are some fundamental differences:
- **Immutable in nature**: Each dataframe once formed, cannot be changed. Transformations used to effect changes to an existing dataframe actually results in the creation of a new one.
- **Lazy Evaluation**: A transformation is not performed until an action is called.
- **Distributed**: Data are stored in distributed fashion on multiple virtual compute units.

#### Libraries and Spark Instantiation
~~~ Python
from pyspark.sql import SparkSession, functions as F, Row
from pyspark.sql.types import *
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext # Get underlying spark context
~~~

#### Creating DataFrames
- Reading from data sources
~~~ Python
df_csv = spark.read.csv('/path/to/yourcsv.csv', header=True,
      inferSchema=True, sep=',', encoding='UTF-8')
df_par = spark.read.parquet('/path/to/parquet/folders')
~~~

- Reading From RDD/Pandas Dataframe using **SparkSession.createDataFrame()**
~~~ python
rdd = sc.parallelize([Row(name='Tom', age=23),
      Row(name='David', age=40)])
df_rdd = spark.createDataFrame(rdd)
import pandas as pd
pandasDF = pd.DataFrame({'name'= ['Tom', 'David'], 'age': [23, 40]})
df_pd = spark.createDataFrame(pandasDF)
~~~

- Programmatically using **SparkSession.createDataFrame()**
~~~ python
my_list = [['Tom', 23], ['David', 40]]
df = spark.createDataFrame(my_list, ['name', 'age'])
~~~

#### Inspecting DataFrames
- Print
  - **df.show(n, truncate=True)** : Print content of df. Optional n and truncate parameters to customise rows printed and truncation
  - **df.printSchema()** : Print schema of df
  - **df.describe().show()** : Print summary statistics (mean, std, min, max, count).
  - **df.explain()** : Print logical and physical plans


- Get Values
  - **df.count()** : Return row count of df
  - **df.columns** : Return list of column names of df
  Note: df.count(), len(df.columns) to replace shape function in pandas
  - **df.take(n)** : Return first n rows of df
  - **df.select('colname').distinct().count()** : Get number of distinct values in column='colname'
  - **df.dtypes** : Return list of column types of df
  - **df.crosstab('Age', 'Gender').show()** : Calculate pair wise frequency of two categorical columns.

#### Missing & Replacing Values
- **df.dropna()** : drop rows with missing values. Parameters include:
  - how = 'any' or 'all'. 'Any' will drop rows with any missing value. 'All' will only drop rows where all fields are missing.
  - thresh = int. Drop rows with non-null values less than thresh. Overwrites how parameter.
  - subset - optional list of column names to consider.


- **df.fillna(value, subset)**
  - value:
    - Dictionary with key the column name and value, the value to fill for that column.
    - A value (int, float, string) for all columns
  - subset: optional list of column names to perform fillna operation.


- **df.na.replace(10, 20)** : replace 10 with 20

#### Filter Rows
- df.filter(df['age']>24).show() # retain only rows where age is >24.
- df.filter(df['age'].isNotNull()) # Select only rows which are not null.

#### Data Selection
 - **Select**
   - **df.select("firstName", "lastName").show()** : select columns
   - **df.select(df['age'] > 24).show()** : select rows where age > 24
   - **df.select(df['firstName'], df['age'] + 1)** : select with derived values
 - **When**
   - **df.select("firstName", F.when(df['age'] > 30, 1).otherwise(0)).show()**
   Select column firstName and derived column with 1s if age is >30 and 0s otherwise.
 - **Between**
   - **df.select(df['age'].between(22, 24)).show()** # show rows where 22<=age<=24
 - **Startswith-Endswith**
   - **df.select("firstName", df['lastName'].startswith("Sm")).show()**
   - **df.select(df['lastName'].endswith("th")).show()**
 - **Substring**
   - **df.select(df['firstName'].substr(1, 3).alias("name")).collect()**
 - **Like**
   - **df.select("firstName", df["lastName"].like("Smith")).show()** : Show *firstName*, and boolean column with value=True for rows where lastName = "Smith" and False otherwise.
 - **rlike**: like using regex patterns
   - **df.select("firstName", df['lastName'].rlike("^Smith")).show()**: show firstname and boolean column with True if lastName starts with Smith.
 - **Isin**
    - **df[df["firstName"].isin(['Jane', 'Boris'])].collect()**

#### Add, Rename & Drop Columns
- **withColumn** to add column
- **withColumnRenamed** to update column name
- **drop** to remove column
~~~ Python
# Add Column
df = df.withColumn('city', df.address.city) \
      .withColumn('postalCode', df.address.postcode) \
      .withColumn('state', df.address.state) \
      .withColumn('telePhoneNumber', explode(df.phoneNumber.number)) \
      .withColumn('telephoneType', explode(df.phoneNumber.type))
# Rename Column
df = df.withColumnRenamed('telePhoneNumber', 'phoneNumber')
# Remove Column
df = df.drop("address", "phoneNumber")
~~~

#### Remove Duplicates
- **DataFrame.select('Age', 'Gender').dropDuplicates().show()** : Get unique ('Age', 'Gender') tuples.

#### GroupBy
- **df.groupBy("age").count().show()** : Show row count for each age group.

#### Sort/ OrderBy
- **df.sort("age", ascending=True).collect()** : sort by age in ascending order
- **df.orderBy(['age', 'city'], ascending=[True, False]).collect()**: sort by age in ascending, then city in descending.


#### Repartitioning
- df.repartition(10).rdd.getNumPartitions() # Split to 10 partitions
- df.coalesce(1).rdd.getNumPartitions() # Merge to 1 partition


#### Register DataFrame as View for SQL Type Queries
- Register DataFrame as Views
  - df.createGlobalTempView("people")
  - df.createTempView("customer")
  - df.createOrReplaceTempView("customer")
- Query Views
  - df5 = spark.sql("SELECT * FROM customer").show()
  - peopledf2 = spark.sql("SELECT * from global_temp.people").show()

#### Convert to Other Data Structure
- rdd1 = df.rdd # Convert to RDD.
- df.toJSON() # Convert to JSON
- df.toPandas() # convert to pandas dataframe

#### Write to Files
- df.write.save("nameAndCity.parquet")
- df.select("firstName", "age").write.save("namesAndAges.json", format="json")


#### Stop SparkSession
- spark.stop()

### Create Resilient Distributed Dataset (RDD)
- pyspark.SparkContext instance can connect to spark cluster and are used to create RDDs using two methods:
  - **parallelize(iterable)**: Distribute a local python collection (e.g. list, tuple, set, dictionary (only keys of dictionary are included as elements of RDD)) to return an RDD.
  - **textFile('path/to/text/csv/file')**: Each line in file is read as string element of returned RDD. Generally postprocessed using map function to cast to numerics for analytics
    - **collect()**: Action to retrieve ALL elements of RDD. Return type is not RDD.
    - **take(n)**: Action to retrieve first n elements of RDD. Return type is not RDD.

~~~ python
# Code Example
from pyspark import SparkContext, SparkConf

sc = SparkContext()
spark = SparkSession(sparkContext=sc)

rdd = sc.parallelize([1, 2, 3])
rdd.collect()

str_rdd = sc.textFile('/path/to/text/csv/file')
str_rdd.take(5)
~~~

### Create Spark DataFrames
- SparkSession can create spark dataframes from **file, RDDs, pandas dataframes, list objects** using the following method:
  - **SparkSession.createDataFrame(object)**: returns a spark dataframe

~~~python
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

sc = SparkContext()
spark = SparkSession(sparkContext=sc)

# From File
df = spark.read.csv('/path/to/csv.csv',
                    header=True,
                    inferSchema=True,
                    sep=',',
                    encoding='UTF-8')
# OR
df = spark.read.format("csv") \
      .option("header", "true") \
      .option("inferSchema", "true") \
      .load('/path/to/csv.csv')

# From RDD
rdd = sc.parallelize([Row(name='Tom', age=23),
                      Row(name='David', age=40)])
df = spark.createDataFrame(rdd)

# From Pandas DataFrame
import pandas as pd
pandasDF = pd.DataFrame({'name'= ['Tom', 'David'], 'age': [23, 40]})
df = spark.createDataFrame(pandasDF)

# From Nested list/Iterable
my_list = [['Tom', 23], ['David', 40]]
df = spark.createDataFrame(my_list, ['name', 'age'])
# Note: if supplied header list falls short of column number, '_colnum' will be used

# Show ((Col_1_name, Col_1_type) ... (Col_n_name, Col_n_type))
df.dtypes
~~~

### RDD and DataFrame Interconversions
- DataFrame to RDD
  - RDD objects have many useful **mapping functions (e.g. map, mapValues, flatMap, flatMapValues)** to manipulate the underlying data.
  - We can get the underlying rdd object from a spark dataframe using its **.rdd** attribute
  - Each element in the returned RDD is a pyspark.sql.Row object. A Row is a list of key-value pairs. E.g. [Row=('name'='Tom', 'age'=23), Row ...]
<br>
- RDD to DataFrame
  - use the SparkSession.createDataFrame(rdd) function. Every element in the RDD has to be a Row object.

~~~ Python
# DataFrame to RDD
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Row
sc = SparkContext()
spark = SparkSession(sparkContext=sc)

mtcars = spark.read.csv('/FileStore/tables/mtcars.csv',
                        header=True, inferSchema=True)
mtcars.rdd.show(2) # show first 5 rows of rdd

# Out[1]:
# [Row(_c0='Mazda RX4', mpg=21.0, cyl=6, disp=160.0, hp=110, drat=3.9, wt=2.62, qsec=16.46, vs=0, am=1, gear=4, carb=4),
#  Row(_c0='Mazda RX4 Wag', mpg=21.0, cyl=6, disp=160.0, hp=110, drat=3.9, wt=2.875, qsec=17.02, vs=0, am=1, gear=4, carb=4)]

# RDD to DataFrame with manipulation
rdd_raw = sc.textFile('/FileStore/tables/mtcars.csv')
rdd_raw.take(2) # each element is a line string
# Out[2]:
# [',mpg,cyl,disp,hp,drat,wt,qsec,vs,am,gear,carb',
#  'Mazda RX4,21,6,160,110,3.9,2.62,16.46,0,1,4,4']

list_of_list = rdd_raw.map(lambda x: x.split(',')) # returns a new RDD
list_of_list.show(2)
# Note: collect()/take(n).first() returns a list and not an RDD

# Out[3]:
# [['','mpg','cyl','disp','hp','drat','wt','qsec','vs','am','gear','carb'],
# ['Mazda RX4','21','6','160','110','3.9','2.62','16.46','0','1','4','4']]

header = list_of_list.first() # header is list containing first row
# Note: first() returns first element ['','mpg','cyl','disp','hp','drat','wt','qsec','vs','am','gear','carb']
# take(1) returns a list containing first element i.e. nested list
# [['','mpg','cyl','disp','hp','drat','wt','qsec','vs','am','gear','carb']]
header[0] = 'model' # substitute '' with 'model'

str_data = list_of_list.filter(lambda x: x[1]!='mpg')
data = str_data.map(lambda x: list(x[0]) + list(map(float, x[1:])))
# Note: map returns a map object, wrapping with list returns elements in list
# Out[4]:
# [['Mazda RX4',21.0,6.0,160.0,110.0,3.9,2.62,16.46,0.0,1.0,4.0,4.0]
# []...]

def map_rows(header, values):
    data_dict = dict(zip(header, values))
    return Row(**data_dict) # ** dict expands {'key': 'value'} to key='value'

final_rdd = data.map(lambda x: map_rows(header, x))
# Out[5]:
# [Row(model='Maxda RX4', mpg='21.0' ...) ...]
mtcars_df = spark.createDataFrame(final_rdd) # Method 1: rdd to DataFrame
mtcars_df = final_rdd.toDF() # Method 2: rdd to DataFrame
# list of colnames is optional parameter to toDF() as info already embedded
# in each Row in this case.
mtcars_df.show(5)
~~~

### Merge/Split Columns Using RDD Map Function
- We can select different sublist of columns for merging to form new columns

~~~ Python
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Row

spark = SparkSession(sparkContext=SparkContext())
mtcars_df = spark.read.csv(path='/FileStore/tables/mtcars.csv',
                           header=True, inferSchema=True,
                           encoding='UTF-8')
header = mtcars_df.columns
header[0] = 'model'
mtcars_df = mtcars_df.rdd.toDF(header) # substitute new header for new DF

# We can regroup values and reassign names to them
mtcars_rdd = mtcars_df.rdd.map(lambda x: Row(model=x[0], values=x[1:]))
# Out[1]:
# [Row(model='Mazda RX4', values=(21.0, 6, 160.0, 110, 3.9, 2.62, 16.46, 0, 1, 4, 4)),
#  Row(model='Mazda RX4 Wag', values=(21.0, 6, 160.0, 110, 3.9, 2.875, 17.02, 0, 1, 4, 4))...]
mtcars_df = spark.createDataFrame(mtcars_rdd)
mtcars_df.show(5, truncate=False)
# +-----------------+-----------------------------------------------------+
# |model            |values                                               |
# +-----------------+-----------------------------------------------------+
# |Mazda RX4        |[21.0, 6, 160.0, 110, 3.9, 2.62, 16.46, 0, 1, 4, 4]  |
# |Mazda RX4 Wag    |[21.0, 6, 160.0, 110, 3.9, 2.875, 17.02, 0, 1, 4, 4] |

# Split and Remerge Values column
mtcars_rdd_2 = mtcars_df.rdd.map(lambda x: Row(model=x[0],
                                           x1=x[1][:4],
                                           x2=x[1][4:]))
# convert RDD back to DataFrame
mtcars_df_2 = spark.createDataFrame(mtcars_rdd_2)
mtcars_df_2.show(5, truncate=False)
# +-----------------+---------------------+--------------------------------+
# |model            |x1                   |x2                              |
# +-----------------+---------------------+--------------------------------+
# |Mazda RX4        |[21.0, 6, 160.0, 110]|[3.9, 2.62, 16.46, 0, 1, 4, 4]  |
# |Mazda RX4 Wag    |[21.0, 6, 160.0, 110]|[3.9, 2.875, 17.02, 0, 1, 4, 4] |
~~~

### Common RDD Map Functions
- **map(function)**: Applies each element of the RDD.
- **mapValues(function)**: Elements has to be in key-value pair. Function applies to value part of each element.
- **flatMap(function)**: Function is applied to each element of an RDD before flattening the results. We can simply use this function to flatten elements of an RDD without extra operation on each elements.
- **flatMapValues(function)**: flatMapValues requires each element in RDD to have key/value pair structure. It applies function to each value part of each RDD element before flattening results.

~~~ python
from pyspark import SparkContext
sc = SparkContext()

# create text rdd using textFile
raw_rdd = sc.textFile('/FileStore/tables/mtcars.csv')
raw_rdd.take(3)
# [',mpg,cyl,disp,hp,drat,wt,qsec,vs,am,gear,carb',
#  'Mazda RX4,21,6,160,110,3.9,2.62,16.46,0,1,4,4',
#  'Mazda RX4 Wag,21,6,160,110,3.9,2.875,17.02,0,1,4,4']

# Create dataframe using map functions
split_rdd = raw_rdd.map(lambda x: x.split(','))
split_rdd.take(2)
# [['', 'mpg','cyl','disp','hp','drat','wt','qsec','vs','am','gear','carb'],
#  ['Mazda RX4', '21', '6', '160', '110', '3.9', '2.62', '16.46', '0', '1', '4', '4']]
header = split_rdd.first()
# ['', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
data_str = split_rdd.filter(lambda x: x!=header) # filter to omit header
data_str.take(2)
# [['Mazda RX4','21','6','160','110','3.9','2.62','16.46','0','1','4','4'],
#  ['Mazda RX4 Wag','21','6','160','110','3.9','2.875','17.02','0','1','4','4']]

# convert string to numeric
data = data_str.map(lambda x: [x[0]] + list(map(float, x[1:])))
# [['Mazda RX4',21.0,6.0,160.0,110.0,3.9,2.62,16.46,0.0,1.0,4.0,4.0],
#  ['Mazda RX4 Wag',21.0,6.0, 160.0, 110.0, 3.9, 2.875, 17.02, 0.0, 1.0, 4.0, 4.0]]

df = spark.createDataFrame(data, header)
df = df.withColumnRenamed('', 'model') # we rename here rather than set
# header[0]='model' earlier because it will affect filter to form data_str
# due to lazy execution.
df.show()
# +-------------------+----+---+-----+-----+----+-----+-----+---+---+----+----+
# |              model| mpg|cyl| disp|   hp|drat|   wt| qsec| vs| am|gear|carb|
# +-------------------+----+---+-----+-----+----+-----+-----+---+---+----+----+
# |          Mazda RX4|21.0|6.0|160.0|110.0| 3.9| 2.62|16.46|0.0|1.0| 4.0| 4.0|
# |      Mazda RX4 Wag|21.0|6.0|160.0|110.0| 3.9|2.875|17.02|0.0|1.0| 4.0| 4.0|
# |         Datsun 710|22.8|4.0|108.0| 93.0|3.85| 2.32|18.61|1.0|1.0| 4.0| 1.0|

# Using mapValues()
import numpy as np
kv_data = data.map(lambda x: (x[0], x[1:]))
avg_res = kv_data.mapValues(lambda x: np.mean(x)) # x here refers to the values only
avg_res.take(2)
# [('Mazda RX4', 29.90727272727273),
#  ('Mazda RX4 Wag', 29.981363636363639)]

# Using flatMap() # flatten items by 1 level
data.take(2)
# [['Mazda RX4',21.0,6.0,160.0,110.0,3.9,2.62,16.46,0.0,1.0,4.0,4.0],
#  ['Mazda RX4 Wag',21.0,6.0, 160.0, 110.0, 3.9, 2.875, 17.02, 0.0, 1.0, 4.0, 4.0]]
data.flatMap(lambda x: x).take(15)
# ['Mazda RX4',21.0,6.0,160.0,110.0,3.9,2.62,16.46,0.0,1.0,4.0,4.0,'Mazda RX4 Wag',
#  21.0,6.0]

# flatMapValues()
# It applies function to each value part of each RDD element before flattening results.
# For example, my raw data looks like below. But I would like to transform the data so that it has three columns: the first column is the sample id; the second the column is the three types (A,B or C); the third column is the values.
# id  A	  B	  C
# 1	  23	18	32
# 2	  18	29	31
# 3	  34	21	18

my_data = [(1, [23, 18, 32]),
           (2, [18, 29, 31]),
           (3, [34, 21, 18])]
my_rdd = sc.parallelize(my_data)
my_rdd.flatMapValues(lambda x: list(zip('ABC', x))).take(5)
# [(1, ('A', 23)),
#  (1, ('B', 28)),
#  (1, ('C', 32)),
#  (2, ('A', 18)),
#  (2, ('B', 29))]
~~~

### Aggregate Functions
- **aggregate(accum, seqOp, combOp)**: merge results to yield single value.
- **aggregateByKey(accum, seqOp, combOp)**: merge results by key.
  - **accum** is data container serving accumulator role. It has common data structure as returned values from seqOp function.
  - **seqOp** is a function that takes two arguments: first argument is the accum and second argument is an element from the RDD. accum gets updated with returned value after every run.
  - **combOp** is a function that combines accums across partitions. It takes two final accums from different partitions as arguments and returns the combined value.

~~~ Python
# calculates squared_diff from mean of mpg and disp using aggregate function
mtcars_df = spark.read.csv('FileStore/tables/mtcars.csv',
      inferSchema=True, header=True).select(['mpg', 'disp'])
mtcars_df.take(3)
# [Row(mpg=21.0, disp=160.0),
#  Row(mpg=21.0, disp=160.0),
#  Row(mpg=22.8, disp=108.0)]
mpg_mean = mtcars_df.select('mpg').rdd.map(lambda x: x[0]).mean()
dis_mean = mtcars_df.select('disp').rdd.map(lambda x: x[0]).mean()
print('mpg mean = ', mpg_mean, '; ' 'disp mean = ', disp_mean)
# mpg mean =  20.090625000000003 ; disp mean =  230.721875

# initial tuple values are containers for squared_diff_from_mean of mpg and displacement
accum = (0, 0)
# accumulator and element in partition
seqOp = lambda a, e: (a[0] + (e[0] - mpg_mean)**2, a[1] + (e[1] - mpg_mean)**2)
# Combine partition x and y
combOp = lambda px, py: (px[0] + py[0], px[1] + py[1])

sqdiff_mpg, sqdiff_disp = mtcars_df.rdd.aggregate(accum, sepOp, combOp).collect()

# aggregateByKey Example
iris_rdd = sc.textFile('FileStore/tables/iris.csv', use_unicode=True)
iris_rdd.take(2)
# ['sepal_length,sepal_width,petal_length,petal_width,species',
#  '5.1,3.5,1.4,0.2,setosa']
iris_rdd_2 = iris_rdd.map(lambda x: x.split(',')).\
    filter(lambda x: x[0] != 'sepal_length').\
    map(lambda x: (x[-1], [*map(float, x[:-1])]))
iris_rdd_2.take(5)
# [('setosa', [5.1, 3.5, 1.4, 0.2]),
#  ('setosa', [4.9, 3.0, 1.4, 0.2]),
#  ('setosa', [4.7, 3.2, 1.3, 0.2]]

accum = (0, 0) # initial tuple values are containers for variance of mpg and displacement
seqOp = lambda a, e: (a[0] + (e[0])**2, a[1] + (e[1])**2)
combOp = lambda px, py: (px[0] + py[0], px[1] + py[1])

iris_rdd_2.aggregateByKey(accum, seqOp, combOp)
# [('versicolor', (1774.8600000000001, 388.47)),
#  ('setosa', (1259.0899999999997, 591.2500000000002)),
#  ('virginica', (2189.9000000000005, 447.33))]
~~~

### Convert Continuous Variables to Categories
- **pyspark.ml.feature.binarizer**: convert to binary based on threshold.
- **pyspark.ml.feature.bucketizer**: convert to categories given split points, n+1 split points for n categories.

~~~ Python
import numpy as np
import pandas as pd
np.random.seed(seed=1234)
pdf = pd.DataFrame({
        'x1': np.random.randn(10),
        'x2': np.random.rand(10)*10
    })
np.random.seed(seed=None)
df = spark.createDataFrame(pdf)
df.show()
# +--------------------+------------------+
# |                  x1|                x2|
# +--------------------+------------------+
# | 0.47143516373249306| 6.834629351721363|
# | -1.1909756947064645| 7.127020269829002|
# |  1.4327069684260973|3.7025075479039495|
# | -0.3126518960917129| 5.611961860656249|
# | -0.7205887333650116| 5.030831653078097|
# |  0.8871629403077386|0.1376844959068224|
# |  0.8595884137174165| 7.728266216123741|
# | -0.6365235044173491| 8.826411906361166|
# |0.015696372114428918| 3.648859839013723|
# | -2.2426849541854055| 6.153961784334937|
# +--------------------+------------------+
from pyspark.ml.feature import Binarizer, Bucketizer
# threshold = 0 for binarizer
binarizer = Binarizer(threshold=0, inputCol='x1', outputCol='x1_new')
# provide 5 split points to generate 4 buckets
bucketizer = Bucketizer(splits=[0, 2.5, 5, 7.5, 10], inputCol='x2', outputCol='x2_new')
# pipeline stages
from pyspark.ml import Pipeline
stages = [bucketizer, binarizer]
pipeline = Pipeline(stages=stages)
# fit the pipeline model and transform the data
pipeline.fit(df).transform(df).show()
# +--------------------+------------------+------+------+
# |                  x1|                x2|x2_new|x1_new|
# +--------------------+------------------+------+------+
# | 0.47143516373249306| 6.834629351721363|   2.0|   1.0|
# | -1.1909756947064645| 7.127020269829002|   2.0|   0.0|
# |  1.4327069684260973|3.7025075479039495|   1.0|   1.0|
# | -0.3126518960917129| 5.611961860656249|   2.0|   0.0|
# | -0.7205887333650116| 5.030831653078097|   2.0|   0.0|
# |  0.8871629403077386|0.1376844959068224|   0.0|   1.0|
# |  0.8595884137174165| 7.728266216123741|   3.0|   1.0|
# | -0.6365235044173491| 8.826411906361166|   3.0|   0.0|
# |0.015696372114428918| 3.648859839013723|   1.0|   1.0|
# | -2.2426849541854055| 6.153961784334937|   2.0|   0.0|
# +--------------------+------------------+------+------+
~~~

### Row Selection
