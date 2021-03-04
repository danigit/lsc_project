import sys
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

# file that analysis the distribution of the available data
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

def write_on_file(data, file_name):
    """Function that write the data passed as parameter on the file passed as parameter as well"""
    data.coalesce(1).write.csv("analysis/" + file_name + ".csv")


def collect_data(data, column1, column2):
    """Function that select from the data passed as parameter the columns passed as parameter as well"""
    return data.select(column1, column2)


def analyse_data(questions, answers, tags):
    """Function that analysis the available data to understand it better."""

    # getting the number of questions that have been made for each year
    questions_per_year = questions \
        .select(F.year('CreationDate').alias('creation_year')) \
        .groupby('creation_year') \
        .count()

    write_on_file(collect_data(questions_per_year, 'creation_year', 'count'), 'questions_distribution_per_year')

    # getting the number of questions that have been made for each month
    questions_per_month = questions \
        .select(F.month('CreationDate').alias('creation_month')) \
        .groupby('creation_month') \
        .count() \
        .sort(F.col('creation_month').asc())

    write_on_file(collect_data(questions_per_month, 'creation_month', 'count'), 'questions_distribution_per_month')

    # getting the number of answered questions for each month
    questions_and_answers = questions.join(answers, questions.Id == answers.ParentId)

    # caching the table because I have to use it later
    questions_and_answers.cache()

    answered_questions_per_month = questions_and_answers\
        .select(questions.Id, questions.CreationDate) \
        .dropDuplicates() \
        .select(F.month('CreationDate').alias('creation_month')) \
        .groupby('creation_month') \
        .count() \
        .sort(F.col('creation_month').asc())

    write_on_file(collect_data(answered_questions_per_month, 'creation_month', 'count'), 'answered_questions_per_month')

    # getting the tags associated with the questions, and for each tag in how many questions has been used
    questions_and_tags = questions.join(tags, questions.Id == tags.Id)

    # caching the data because I have to use it later
    questions_and_tags.cache()

    questions_tags = questions_and_tags \
        .groupby(tags.Tag) \
        .count() \
        .sort(F.col('count').desc())

    write_on_file(collect_data(questions_tags, 'Tag', 'count'), 'tags_frequency')

    # getting the number of answered questions
    answered_questions = questions_and_answers \
        .select(questions.Id) \
        .dropDuplicates()

    # getting the tags that have been used for the answered questions, and in how many answered questions the tag
    # has been used
    answered_questions_and_tags = answered_questions.join(tags, answered_questions.Id == tags.Id)

    # caching the data because I have to use it later
    answered_questions_and_tags.cache()

    answered_tags = answered_questions_and_tags \
        .groupby('Tag') \
        .count() \
        .sort(F.col('count').desc())

    write_on_file(collect_data(answered_tags, 'Tag', 'count'), 'tag_frequency_answered_questions')

    # getting the tags for the most upvoted questions
    upvoted_tags = questions_and_tags \
        .groupby('Tag') \
        .agg(F.sum("Score").alias('score_sum')) \
        .sort(F.col('score_sum').desc())

    write_on_file(collect_data(upvoted_tags, 'Tag', 'score_sum'), 'tag_frequency_by_upvotes')

    # getting the number of tags for the answered questions
    answered_tags_number = answered_questions_and_tags \
        .groupby(answered_questions.Id) \
        .agg(F.count(tags.Tag).alias('tags_number')) \
        .groupby(F.col('tags_number')) \
        .count() \
        .sort(F.col('tags_number').desc())

    write_on_file(collect_data(answered_tags_number, 'tags_number', 'count'), 'number_of_tags_for_each_question')

    # getting the not answered questions
    not_answered_questions = questions.join(answers, questions.Id == answers.ParentId, how='left') \
        .filter(F.col('ParentId').isNull()) \
        .select([questions.Id])

    # getting the number of tags for not answered questions
    not_answered_tags_number = not_answered_questions.join(tags, not_answered_questions.Id == tags.Id) \
        .groupby(not_answered_questions.Id) \
        .agg(F.count(tags.Tag).alias('tags_number')) \
        .groupby(F.col('tags_number')) \
        .count() \
        .sort(F.col('tags_number').desc())

    write_on_file(collect_data(not_answered_tags_number, 'tags_number', 'count'), 'number_of_tags_for_each_answered_question')

    # getting the number of tags for the most upvoted questions
    answered_tags_upvotes = questions_and_tags \
        .groupby([questions.Id, questions.Score]) \
        .agg(F.count(tags.Tag).alias('tag_count')) \
        .sort(F.col('Score').desc())

    write_on_file(collect_data(answered_tags_upvotes, 'Score', 'tag_count'), 'number_of_tags_for_upvoted_questions')

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

    # loading the questions
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

    # loading the answers
    answers = load_data(answers_path, answers_schema)

    answers.cache()

    # creating the tags schema
    tags_schema = StructType([
        StructField('Id', IntegerType(), False),
        StructField('Tag', StringType(), True)
    ])

    # loading the tags
    tags = load_data(tags_path, tags_schema)

    tags.cache()

    # analyzing the data
    analyse_data(questions, answers, tags)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: analysis_local <files>", file=sys.stderr)
        exit(-1)

    # getting the start execution time
    start = time.time()

    # getting the csv files 
    questions_path = sys.argv[1]
    answers_path = sys.argv[2]
    tags_path = sys.argv[3]

    # creating the spark sessionS
    spark = SparkSession.builder.appName('Stackoverflow analisys with spark').getOrCreate()

    main()

    # getting the end execution time
    end = time.time()

    # computing the total execution time
    print("The total execution time is: ", end - start)
