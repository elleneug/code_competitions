import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType

#TODO аннотация
@pandas_udf(StringType())
def card_number_mask(s: pd.Series) -> pd.Series:
    # Mask for card numbers
    return s.apply(lambda x: x[:4] + 'X' * (len(x) - 8) + x[-4:] if len(x) == 16 else x)


if __name__ == "__main__":
    spark = SparkSession.builder.appName('PySparkUDF').getOrCreate()
    df = spark.createDataFrame([(1, "4042654376478743"), (2, "4042652276478747")], ["id", "card_number"])
    df.show()
    dfr = df.withColumn("hidden", card_number_mask("card_number"))
    dfr.show(truncate=False)
