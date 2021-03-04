import re
import string
import sys
import time

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark import StorageLevel as stl
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


@F.udf(returnType=StringType())
def remove_htm_tags(text):
    """User defined function that removes all the html tags from the string passed as parameter.
    The decorator allows to this function to be used as a built in pispark function."""

    # declaring the regular expression for finding html code
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


@F.udf(returnType=StringType())
def remove_stopwords(text):
    """Function that remove the stopwards from the string passed as parameter.
    The language is english.
    The decorator allows to this function to be used as a built in pispark function."""

    return ' '.join([word for word in text.split() if word not in cached_stop_words])


@F.udf(returnType=StringType())
def remove_punctuation(text):
    """Function that remove punctuation from the string passed as parameter
    The decorator allows to this function to be used as a built in pispark function."""

    return text.translate(str.maketrans('', '', string.punctuation))

@F.udf(returnType=StringType())
def stem(text):
    """Function that stem the string passed as parameter
    The decorator allows to this function to be used as a built in pispark function"""

    words = text.split(' ')
    words = list(map(lambda word: stemmer.stem(word), words))
    return ' '.join(map(str, words))


def load_data(path, schema):
    """Function that load the data from the files passed as parameter, using the schema passed
    as parameter as well."""
    return spark \
        .read \
        .option('multiline', True) \
        .option('escape', "\"") \
        .option('header', 'true') \
        .schema(schema) \
        .csv(path)


def create_features(data_set, input_col, number_of_features=10000, index_label=True):
    """Function that creates the features from the row data. Transform the text in associated numbers
    so that can be use as classification dataset."""

    # splitting the text in tokens
    tokenizer = Tokenizer(inputCol=input_col, outputCol="body_words")

    # transforming the tokens (word) in numbers using Inverse Document Frequency
    hashing_tf = HashingTF(inputCol="body_words", outputCol="raw_body_features", numFeatures=number_of_features)
    idf = IDF(inputCol="raw_body_features", outputCol="features", minDocFreq=0)

    # using a pipeline to apply all the above operations
    if index_label:
        label_string_idx = StringIndexer(inputCol="Tag", outputCol="label")
        pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf, label_string_idx])
    else:
        pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf])

    pipeline_fit = pipeline.fit(data_set)

    return pipeline_fit.transform(data_set)


def run_tag_prediction(questions, tags):
    """Function that apply the computation in order to obtain the tag prediction"""

    # getting the tags associated with the questions
    questions_tags = questions \
        .join(tags, ['Id']) \
        .filter(F.col('Score') >= score_threshold) \
        .drop('Id', 'OwnerUserId', 'CreationDate', 'ClosedDate', 'Score')

    # storing the result for future use
    questions_tags.cache()

    # getting the tags with the highest frequency
    highest_frequency_tags = questions_tags.groupBy('Tag') \
        .count() \
        .sort(F.col('count').desc()) \
        .select('Tag') \
        .limit(tags_frequency_threshold).rdd \
        .flatMap(lambda tag: tag).collect()

    # getting the questions that have relevant tags
    questions_with_relevant_tags = questions_tags \
        .filter(F.col('Tag').isin(highest_frequency_tags))

    questions_with_relevant_tags.cache()

    # processing the data by removing punctuation, removing html, removing stopwords and stemming the text
    preprocessed_data = questions_with_relevant_tags \
        .select(stem(remove_stopwords(remove_punctuation(F.lower(F.col('Title'))))),
                stem(remove_stopwords(remove_punctuation(remove_htm_tags(F.lower(F.col('Body')))))), 'Tag') \
        .withColumnRenamed('stem(remove_stopwords(remove_punctuation(lower(Title))))', 'Title') \
        .withColumnRenamed('stem(remove_stopwords(remove_punctuation(remove_htm_tags(lower(Body)))))', 'Body')

    # getting only the data we need
    training_set = preprocessed_data \
        .withColumn('Body',
            F.concat(
                F.col('Title'),
                F.lit(' '),
                F.col('Body')
            )) \
        .drop('Title')

    # creating the features
    features_data = create_features(training_set, 'Body')

    # splitting the data in training and test
    (trainingData, testData) = features_data.randomSplit([0.8, 0.2], seed=100)

    # applying logistic regression on training data
    logistic_regression = LogisticRegression(maxIter=10, regParam=0.01, elasticNetParam=0.3)

    lr_model = logistic_regression.fit(trainingData)

    # taking the prediction
    predictions = lr_model.transform(testData)

    # evaluating the accuracy
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

    # creating the parameters for the cross validation
    parameter_grid = ParamGridBuilder() \
        .addGrid(logistic_regression.regParam, [0.01]) \
        .addGrid(logistic_regression.elasticNetParam, [0.0, 0.1, 0.2, 0.3, 0.4]) \
        .addGrid(logistic_regression.maxIter, [10]) \
        .build()

    # applying the cross validation
    cv = CrossValidator(estimator=logistic_regression, estimatorParamMaps=parameter_grid, evaluator=evaluator, numFolds=5)
    cv_model = cv.fit(trainingData)
    best_model = cv_model.bestModel

    # creating the best prediction
    best_predictions = best_model.transform(testData)
    best_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("Best reg param:", best_model._java_obj.getRegParam())
    print("Best max iter:", best_model._java_obj.getMaxIter())
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    print(f'The accuracy in predicting the tags is: {evaluator.evaluate(predictions)}')
    print(f'The accuracy in predicting the tags using the best model is: {best_evaluator.evaluate(best_predictions)}')



def main():
    # creating the questions schema
    questions_schema = StructType([
        StructField('Id', IntegerType(), False),
        StructField('OwnerUserId', IntegerType(), True),
        StructField('CreationDate', StringType(), True),
        StructField('ClosedDate', StringType(), True),
        StructField('Score', IntegerType(), True),
        StructField('Title', StringType(), True),
        StructField('Body', StringType(), True)
    ])

    # loading the questions data
    questions = load_data(questions_path, questions_schema)
    questions.cache()

    # creating the tags schema
    tags_schema = StructType([
        StructField('Id', IntegerType(), False),
        StructField('Tag', StringType(), True)
    ])

    # loading the tags data
    tags = load_data(tags_path, tags_schema)
    tags.cache()

    # making the prediction
    run_tag_prediction(questions, tags)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: tag_prediction_local <files>", file=sys.stderr)
        exit(-1)

    # getting the time before the execution
    start = time.time()

    # defining the score threshold
    score_threshold = 10

    # defining the tags frequency threshold
    tags_frequency_threshold = 10

    # getting the csv files
    questions_path = sys.argv[1]
    tags_path = sys.argv[2]

    # creating the spark session
    spark = SparkSession.builder.appName('Stackoverflow analisys with spark').getOrCreate()

    # creating the stemmer
    cached_stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')

    main()

    # getting the time after the execution finishes
    end = time.time()

    # computing the execution time
    print(f"The total execution time is {end - start}")