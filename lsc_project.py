# IMPORTING THE REQUIRED LIBRARIES
import re
import string
import sys
import time

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *


@F.udf(returnType=StringType())
def remove_htm_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


@F.udf(returnType=StringType())
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in cached_stop_words])


@F.udf(returnType=StringType())
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


@F.udf(returnType=StringType())
def stem(text):
    words = text.split(' ')
    words = list(map(lambda word: stemmer.stem(word), words))
    return ' '.join(map(str, words))

def load_data(path, schema):
    return spark \
        .read \
        .option('multiline', True) \
        .option('escape', "\"") \
        .option('header', 'true') \
        .schema(schema) \
        .csv(path)


def create_features(data_set, input_col, number_of_features=1000, index_label=True):
    tokenizer = Tokenizer(inputCol=input_col, outputCol="body_words")
    hashing_tf = HashingTF(inputCol="body_words", outputCol="raw_body_features", numFeatures=number_of_features)
    idf = IDF(inputCol="raw_body_features", outputCol="features", minDocFreq=0)

    if index_label:
        label_string_idx = StringIndexer(inputCol="Tag", outputCol="label")
        pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf, label_string_idx])
    else:
        pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf])

    pipeline_fit = pipeline.fit(data_set)

    return pipeline_fit.transform(data_set)


def run_tag_prediction(questions, tags):
    questions_tags = questions \
        .join(tags, ['Id']) \
        .filter(F.col('Score') >= score_threshold) \
        .drop('Id', 'OwnerUserId', 'CreationDate', 'ClosedDate', 'Score')

    questions_tags.cache()

    highest_frequency_tags = questions_tags.groupBy('Tag') \
        .count() \
        .sort(F.col('count').desc()) \
        .select('Tag') \
        .limit(tags_frequency_threshold).rdd \
        .flatMap(lambda tag: tag).collect()

    questions_with_relevant_tags = questions_tags \
        .filter(F.col('Tag').isin(highest_frequency_tags))

    questions_with_relevant_tags.cache()

    preprocessed_data = questions_with_relevant_tags \
        .select(stem(remove_stopwords(remove_punctuation(F.lower(F.col('Title'))))),
                stem(remove_stopwords(remove_punctuation(remove_htm_tags(F.lower(F.col('Body')))))), 'Tag') \
        .withColumnRenamed('stem(remove_stopwords(remove_punctuation(lower(Title))))', 'Title') \
        .withColumnRenamed('stem(remove_stopwords(remove_punctuation(remove_htm_tags(lower(Body)))))', 'Body')

    training_set = preprocessed_data \
        .withColumn('Body',
                    F.concat(
                        F.col('Title'),
                        F.lit(' '),
                        F.col('Body')
                    )) \
        .drop('Title')

    features_data = create_features(training_set, 'Body')

    (trainingData, testData) = features_data.randomSplit([0.7, 0.3], seed=100)
    logistic_regression = LogisticRegression(maxIter=10, regParam=0.1, elasticNetParam=0.3)
    lr_model = logistic_regression.fit(trainingData)

    predictions = lr_model.transform(testData)

    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    print(f'The accuracy in predicting the tags is: {evaluator.evaluate(predictions)}')


def run_answer_prediction(questions, answers):
    questions_answers = questions \
        .join(answers, questions.Id == answers.ParentId) \
        .select(questions.Score.alias('questions_score'),
                questions.Title.alias('questions_title'),
                questions.Body.alias('questions_body'),
                answers.ParentId.alias('label')) \
        .dropDuplicates() \
        .withColumn('label', F.when(F.col('label').isNotNull(), 1).otherwise(0))

    questions_answers.cache()

    answered_questions = questions_answers \
        .filter(questions_answers.label == 1) \
        .sort(questions_answers.questions_score.desc()) \
        .limit(161648)

    questions_without_answers = questions_answers \
        .filter(questions_answers.label == 0)

    # balancing the number of answered questions and the number of not answered questions
    balanced_data = answered_questions.union(questions_without_answers)

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

    features_data = create_features(training_set, 'questions_body', index_label=False)

    final_dataset = features_data.select("features", "label")
    (trainingData, testData) = final_dataset.randomSplit([0.7, 0.3], seed=100)
    logistic_regression = LogisticRegression(maxIter=10, regParam=0.00001, elasticNetParam=0)
    lr_model = logistic_regression.fit(trainingData)

    predictions = lr_model.transform(testData)

    evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
    print(f'The accuracy in predicting the answers is: {evaluator.evaluate(predictions)}')


def write_on_file(data, file_name):
    data.coalesce(1).write.csv("analysis/" + file_name + ".csv")

def collect_data(data, column1, column2):
    return data.select(column1, column2)

def analyse_data(questions, answers, tags):
    questions_per_year = questions \
        .select(F.year('CreationDate').alias('creation_year')) \
        .groupby('creation_year') \
        .count()

    write_on_file(collect_data(questions_per_year, 'creation_year', 'count'), 'questions_distribution_per_year')

    questions_per_month = questions \
        .select(F.month('CreationDate').alias('creation_month')) \
        .groupby('creation_month') \
        .count() \
        .sort(F.col('creation_month').asc())

    write_on_file(collect_data(questions_per_month, 'creation_month', 'count'), 'questions_distribution_per_month')

    questions_and_answers = questions.join(answers, questions.Id == answers.ParentId)
    questions_and_answers.cache()


    answered_questions_per_month = questions_and_answers \
        .select(questions.Id, questions.CreationDate) \
        .dropDuplicates() \
        .select(F.month('CreationDate').alias('creation_month')) \
        .groupby('creation_month') \
        .count() \
        .sort(F.col('creation_month').asc())

    write_on_file(collect_data(answered_questions_per_month, 'creation_month', 'count'), 'answered_questions_per_month')

    questions_and_tags = questions.join(tags, questions.Id == tags.Id)
    questions_and_tags.cache()


    questions_tags = questions_and_tags \
        .groupby(tags.Tag) \
        .count() \
        .sort(F.col('count').desc())

    write_on_file(collect_data(questions_tags, 'Tag', 'count'), 'tags_frequency')

    answered_questions = questions_and_answers \
        .select(questions.Id) \
        .dropDuplicates()

    answered_questions_and_tags = answered_questions.join(tags, answered_questions.Id == tags.Id)
    answered_questions_and_tags.cache()


    answered_tags = answered_questions_and_tags \
        .groupby('Tag') \
        .count() \
        .sort(F.col('count').desc())

    write_on_file(collect_data(answered_tags, 'Tag', 'count'), 'tag_frequency_answered_questions')

    upvoted_tags = questions_and_tags \
        .groupby('Tag') \
        .agg(F.sum("Score").alias('score_sum')) \
        .sort(F.col('score_sum').desc())

    write_on_file(collect_data(upvoted_tags, 'Tag', 'score_sum'), 'tag_frequency_by_upvotes')

    answered_tags_number = answered_questions_and_tags \
        .groupby(answered_questions.Id) \
        .agg(F.count(tags.Tag).alias('tags_number')) \
        .groupby(F.col('tags_number')) \
        .count() \
        .sort(F.col('tags_number').desc())

    write_on_file(collect_data(answered_tags_number, 'tags_number', 'count'), 'number_of_tags_for_each_question')

    not_answered_questions = questions.join(answers, questions.Id == answers.ParentId, how='left') \
        .filter(F.col('ParentId').isNull()) \
        .select([questions.Id])

    not_answered_tags_number = not_answered_questions.join(tags, not_answered_questions.Id == tags.Id) \
        .groupby(not_answered_questions.Id) \
        .agg(F.count(tags.Tag).alias('tags_number')) \
        .groupby(F.col('tags_number')) \
        .count() \
        .sort(F.col('tags_number').desc())

    write_on_file(collect_data(not_answered_tags_number, 'tags_number', 'count'), 'number_of_tags_for_each_answered_quesion')

    answered_tags_upvotes = questions_and_tags \
        .groupby([questions.Id, questions.Score]) \
        .agg(F.count(tags.Tag).alias('tag_count')) \
        .sort(F.col('Score').desc())

    write_on_file(collect_data(answered_tags_upvotes, 'Score', 'tag_count'), 'number_of_tags_for_upvoted_questions')

    
def main():
    answers_schema = StructType([
        StructField('Id', IntegerType(), False),
        StructField('OwnerUserId', IntegerType(), True),
        StructField('CreationDate', StringType(), True),
        StructField('ParentId', IntegerType(), False),
        StructField('Score', IntegerType(), True),
        StructField('Body', StringType(), True)
    ])

    answers = load_data(answers_path, answers_schema)
    answers.cache()

    questions_schema = StructType([
        StructField('Id', IntegerType(), False),
        StructField('OwnerUserId', IntegerType(), True),
        StructField('CreationDate', StringType(), True),
        StructField('ClosedDate', StringType(), True),
        StructField('Score', IntegerType(), True),
        StructField('Title', StringType(), True),
        StructField('Body', StringType(), True)
    ])

    questions = load_data(questions_path, questions_schema)
    questions.cache()

    tags_schema = StructType([
        StructField('Id', IntegerType(), False),
        StructField('Tag', StringType(), True)
    ])

    tags = load_data(tags_path, tags_schema)

    tags.cache()

    analyse_data(questions, answers, tags)
    run_tag_prediction(questions, tags)
    run_answer_prediction(questions, answers)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: lsc_project <files>")
        exit(-1)

    start = time.time()

    score_threshold = 10
    tags_frequency_threshold = 10

    questions_path = sys.argv[1]
    answers_path = sys.argv[2]
    tags_path = sys.argv[3]

    spark = SparkSession.builder.appName('Stackoverflow analisys with spark').getOrCreate()

    cached_stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    
    main()

    end = time.time()
    print("The total execution time is: ", end - start)