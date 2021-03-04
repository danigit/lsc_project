# import findspark
import re
import string
import sys
import time

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *



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


def create_features(data_set, input_col, number_of_features=1000, index_label=True):
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


def run_answer_prediction(questions, answers):
    """Function that apply the computation in order to obtain the probability of the questions
    of getting an aswer."""

    questions_answers = questions \
        .join(answers, questions.Id == answers.ParentId) \
        .select(questions.Score.alias('questions_score'),
                questions.Title.alias('questions_title'),
                questions.Body.alias('questions_body'),
                answers.ParentId.alias('label')) \
        .dropDuplicates() \
        .withColumn('label', F.when(F.col('label').isNotNull(), 1).otherwise(0))

    questions_answers.cache()

    # getting the same number of answered questions as the number of not answered questions
    answered_questions = questions_answers \
        .filter(questions_answers.label == 1) \
        .sort(questions_answers.questions_score.desc()) \
        .limit(161648)

    # getting the not answered questions
    questions_without_answers = questions_answers \
        .filter(questions_answers.label == 0)

    # balancing the number of answered questions and the number of not answered questions
    balanced_data = answered_questions.union(questions_without_answers)

    # applying NLP tool to prepare the data for the processing
    processed_dataset = balanced_data \
        .select(stem(remove_stopwords(remove_punctuation(F.lower(F.col('questions_title'))))),
                stem(remove_stopwords(remove_punctuation(remove_htm_tags(F.lower(F.col('questions_body')))))), 'label') \
        .withColumnRenamed('stem(remove_stopwords(remove_punctuation(lower(questions_title))))', 'questions_title') \
        .withColumnRenamed('stem(remove_stopwords(remove_punctuation(remove_htm_tags(lower(questions_body)))))',
                           'questions_body')

    training_set = processed_dataset \
        .withColumn('questions_body',
                    F.concat(
                        F.col('questions_title'),
                        F.lit(' '),
                        F.col('questions_body')
                    )) \
        .drop('questions_title')

    # creating the data features
    features_data = create_features(training_set, 'questions_body', index_label=False)

    # creating the training and test data
    final_dataset = features_data.select("features", "label")
    (trainingData, testData) = final_dataset.randomSplit([0.7, 0.3], seed=100)

    # creating the logistic regression model
    logistic_regression = LogisticRegression(maxIter=10, regParam=0.00001, elasticNetParam=0)
    lr_model = logistic_regression.fit(trainingData)

    # making the prediction
    predictions = lr_model.transform(testData)

    # evaluating the result
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")

    # creating the parameters for the cross validation
    parameter_grid = ParamGridBuilder() \
        .addGrid(logistic_regression.regParam, [0.00001, 0.001, 0.01, 0.1]) \
        .addGrid(logistic_regression.elasticNetParam, [0.0, 0.1, 0.3, 0.5]) \
        .addGrid(logistic_regression.maxIter, [10, 20, 40]) \
        .build()

    # applying the cross validation
    cv = CrossValidator(estimator=logistic_regression, estimatorParamMaps=parameter_grid, evaluator=evaluator, numFolds=5)
    cv_model = cv.fit(trainingData)
    best_model = cv_model.bestModel

    # creating the best prediction
    best_predictions = best_model.transform(testData)
    best_evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("Best reg param:", best_model._java_obj.getRegParam())
    print("Best max iter:", best_model._java_obj.getMaxIter())
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    print(f'The accuracy in predicting the answers is: {evaluator.evaluate(predictions)}')
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

    # loading the questions from the csv file
    questions = load_data(questions_path, questions_schema)
    questions.cache()

    # creating the answers schema
    answers_schema = StructType([
        StructField('Id', IntegerType(), False),
        StructField('OwnerUserId', IntegerType(), True),
        StructField('CreationDate', StringType(), True),
        StructField('ParentId', IntegerType(), False),
        StructField('Score', IntegerType(), True),
        StructField('Body', StringType(), True)
    ])

    # loading the answers from the csv file
    answers = load_data(answers_path, answers_schema)
    answers.cache()

    # making the predictions
    run_answer_prediction(questions, answers)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: movie <file>", file=sys.stderr)
        exit(-1)

    # getting the time before the execution
    start = time.time()

    # getting the csv files
    questions_path = sys.argv[1]
    answers_path = sys.argv[2]

    # creating the spark session
    spark = SparkSession.builder.appName('Stackoverflow analisys with spark').getOrCreate()

    # creating the stemmer
    cached_stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')

    main()

    # getting the time after the execution
    end = time.time()

    # computing the execution time
    print(f"The total execution time is: {end - start}")