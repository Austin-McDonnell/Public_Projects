import pyodbc
import logging as logger
import pandas as pd
import numpy as np
import os
from typing import Optional, Union, List, Dict, Tuple

import country_converter as coco
from sqlalchemy import create_engine

from dataclasses import dataclass


class DataBase:

    def __init__(self):
        self.engine = create_engine('sqlite://', echo=False)


@dataclass(order=True)
class MovieRecord:
    director_name: str
    duration: int
    gross: float
    movie_title: str
    title_year: int
    language: str
    country: str
    budget: float
    imdb_score: float
    movie_facebook_likes: int



class MovieData:
    file_name = r'movie_sample_dataset.csv'

    dtypes = {
        'director_name': 'object',
        'duration': 'int64',
        'gross': 'float64',
        'genres': 'object',
        'movie_title': 'object',
        'title_year': 'int64',
        'language': 'object',
        'country': 'object',
        'budget': 'float64',
        'imdb_score': 'float64',
        'actors': 'object',
        'movie_facebook_likes': 'int64'
    }

    static_string_data = {
        'director_name',
        'genres',
        'movie_title',
        'language',
        'country',
        'actors'
    }

    def __init__(self):
        self.movie_data_df: Optional[pd.DataFrame] = None

    def _clean_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Cleans all of the numerical data in the DataFrame; Currently handles filling NaN data as well as converting negative
        numbers in columns that should not be negative into positive numbers
        :param df:
        :return:
        '''
        logger.info('Cleaning Numerical Data')
        # TODO: "title_year" has typos in the data
        # TODO: Not sure if to remove the negative data or coerce it to non-negative
        fill_na_dict = {}
        non_negative_columns = ['duration', 'imdb_score', 'movie_facebook_likes', 'budget', 'title_year']
        for dtype in self.dtypes.items():

            if 'int' in dtype[1]:
                fill_na_dict.update({dtype[0]: int(0)})
            elif 'float' in dtype[1]:
                fill_na_dict.update({dtype[0]: float(0)})

        df.loc[:, non_negative_columns] = df.loc[:, non_negative_columns].apply('abs')

        return df.fillna(fill_na_dict)

    def _clean_string_data(self, df: pd.DataFrame):
        '''
        Cleans string data from the data table: Currently only handles country data cleaning
        :param df: the dirty data table
        :return: DataFrame
        '''
        logger.info('Cleaning String data')
        cc = coco.CountryConverter()
        # strange happening with UK not converting to GBR so had to manually do it
        # Also could not get the country converter to work with Lambda and .assign
        df = df.replace({'country': {'UK': 'United Kingdom'}})
        df['country'] = cc.convert(df.country.to_list(), to='ISO3')
        return df

    @staticmethod
    def _string_column_split(column_series: pd.Series, split_value: str) -> pd.DataFrame:
        '''
        Splits the stting columns into multiple columns to allow for "drill down" like feature on sub-data
        :param column_series: Series value of the column from the main data frame that needs to be split apart
        :param split_value: the separator value in each string
        :return: DataFrame of sub-data
        '''

        logger.info(f'Splitting: {column_series.name} string data into sub-tables')
        column_name = column_series.name

        sub_df = column_series.str.split(split_value, expand=True).fillna(np.nan)
        sub_df.columns = [f'{column_name}_{num}' for num in range(1, len(sub_df.columns) + 1)]

        return sub_df

    def get_unique_values(self, column_name) -> np.ndarray:
        '''
        Returns an array of all the unique values in that particular column; performs somewhat of a drill down on the columns
        "actors" and "genres" where the string data is split apart; Only allowed to get unique values on string/categorical data
        :param column_name: the column name to get unique values from
        :return: numpy array of string values
        '''
        logger.info(f'Getting Unique values from: {column_name}')
        allowed_unique_values = ['director_name', 'genres', 'movie_title', 'language', 'country', 'actors']
        if column_name not in allowed_unique_values:
            raise ValueError(f'Not allowed to get unique values on column: {column_name}')

        # TODO: Make these separate sub-data tables possibly instead of creating from scratch each time
        elif column_name in ['actors', 'genres']:
            sep = ',' if column_name == 'actors' else '|'
            values = self._string_column_split(self.movie_data_df.loc[:, column_name], split_value=sep).values.ravel()
            return np.unique(values[~pd.isna(values)])

        return self.movie_data_df.loc[:, column_name].unique()

    def get_filtered_movie_data(self, filter_dict: Dict) -> pd.DataFrame:
        '''
        Allows the user to pass through a dictionary of Keys as column/header names in the data and values as either a
        list of strings or a single string item of what they want to filter on. This filtering process is cumulative, which
        is why a dictionary was choosen for this argument
        :param filter_dict: The dictionary of column keys and filtered values (values found in the data)
        :return: DataFrame that has been filtered down based on the user given dictionary

        Ex: filter_dict = {'genres': ['Drama', 'Crime']} -> this will return all of the rows that have crime and drama
        values in the genre column
        '''
        logger.info('Performing pandas filter on the data')
        known_fields = self.movie_data_df.columns

        filtered_df = self.movie_data_df.copy()

        # Iterates though the filter dictionary passed and will cumulatively filter down the data based on the passed
        # arguments
        for filter_value in filter_dict.items():
            column_name, value = filter_value[0], filter_value[1]
            if not isinstance(value, list):
                value = [value]
            if column_name not in known_fields:
                raise ValueError(f'The filtered field; {column_name} is not listed in the current data')

            elif column_name in ['actors', 'genres']:
                sep = ',' if column_name == 'actors' else '|'
                filtered_df = filtered_df.loc[self._string_column_split(self.movie_data_df.loc[:, column_name], split_value=sep).isin(value).any(axis=1), :]
            else:
                filtered_df = filtered_df.loc[filtered_df.loc[:, column_name].isin(value)]

        return filtered_df

    def _calculate_metrics(self, df: pd.DataFrame):
        '''
        Calculates performance metrics on each movie title
        :param df: the main DataFrame that the calculations will run through
        :return: DataFrame with the added performance metric columns
        '''
        # Remove dirty data before calculations are to be fully accurate;
        logger.info('Calculating Performance Metrics')
        return (df.assign(profit=lambda x: x.gross - x.budget)
                .assign(profit_to_imdb_ratio=lambda x: x.profit / x.imdb_score)
                )

    def calculate_statistics(self, grouping: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        '''
        Returns the numerical statistics such as mean, median, count, standard deviation
        :param grouping: allows the user to pass single or multiple column header values in the function for grouped statistics
        :return: DataFrame of the stats calculations
        '''
        logger.info('Calculating standard statistics on numerical data')
        if grouping:
            logger.info(f'Grouping by: {grouping}')

            if isinstance(grouping, str):
                grouping = [grouping]

            if not all(True if col in self.movie_data_df.columns else False for col in grouping):
                raise ValueError(f'The grouping values passed are not in the data table header:\n'
                                 f'{set(self.movie_data_df.columns).symmetric_difference(set(grouping))}')
            # This does not work as intended for the sub-level data on genres and actors
            return self.movie_data_df.groupby(grouping).describe().round().stack()
        else:
            # The "50%" Level is equivalent to the median; the mean is equal to the average
            return self.movie_data_df.describe().round()



    def read_movie_data(self) -> pd.DataFrame:
        '''
        Performs the main reading in from the csv file; then cleans and calculates performance metrics on the data
        :return: DataFrame of the entire table
        '''

        # TODO: Need to handle how to account for "Duplicated" Data that is not technically duplicated; need to combine
        # the same title data together
        logger.info('Reading in the raw csv data; cleaning data and performing calculations')
        self.movie_data_df = (pd.read_csv(os.path.join(os.path.dirname(__file__), self.file_name), dtype=self.dtypes)
                              .pipe(self._clean_numeric_data)
                              .pipe(self._clean_string_data)
                              .drop_duplicates()
                              .pipe(self._calculate_metrics)
                              )
        self.movie_data_df.index.name = 'MovieKeyNumber'

        return self.movie_data_df

    def write_to_database(self, data_base: DataBase):
        '''
        Used to write the entire movies DataFrame into an SQL Lite Database
        :param data_base: the DataBase class instance
        :param table_name: The name of the table that will be stored in the database
        :return: None
        '''
        # Need to figure out a way to give each record a KeyNumber
        # self.movie_data_df.index.name = 'MovieKeyNumber'
        logger.info('Writing entire data into table within the database')
        genre_df = self._string_column_split(self.movie_data_df.genres, split_value='|')
        genre_df.index.name = 'MovieKeyNumber'

        (genre_df
         .reset_index()
         .to_sql('Genres', con=data_base.engine, if_exists='replace', index=False))

        actors_df = self._string_column_split(self.movie_data_df.actors, split_value=',')
        actors_df.index.name = 'MovieKeyNumber'

        (actors_df
         .reset_index()
         .to_sql('Actors', con=data_base.engine, if_exists='replace', index=False))

        (self.movie_data_df.drop(['actors', 'genres'], axis=1)
         .reset_index()
         .to_sql('Movies', con=data_base.engine, if_exists='replace', index=False))

    @staticmethod
    def write_record_to_database(record: Union[MovieRecord, List[MovieRecord]], data_base: DataBase, table_name: str):
        '''
        Used to write individual or multiple records of type MoveRecord to the database; utilizes a data class structure
        for more rigid writing; forcing the user to have proper data types and full values
        :param record: the records or list of records to be written
        :param data_base: the DataBase class that stores all of the database information
        :param table_name: The name of the table to append the records to
        :return: None
        '''
        # TODO: Need to update this logic to handle adding new data with unique keys when there may be the same key
        #  already present in the database
        # Does not handle writing to actors and genres tables
        logger.info('Writting records to database')
        df = pd.DataFrame(record)
        # df.index.name = 'MovieKeyNumber'
        df.to_sql(table_name, con=data_base.engine, if_exists='replace')

    def read_from_database(self, sql_string_query: str, data_base: DataBase) -> pd.DataFrame:
        '''
        Reads the data from the database when given an SQL Query
        :param sql_string_query: String formatted sql query
        :param data_base: DataBase Class Instance where the database engine is stored
        :return: pd.DataFrame of the query
        '''
        # This is method below uses the engine and sqlalchemy directly; pd.read_sql is easier and more simple to use
        # records = data_base.engine.execute(sql_string_query).fetchall()
        # return pd.DataFrame.from_records(records)
        logger.info('Reading data from database')
        return pd.read_sql(sql_string_query, con=data_base.engine)


if __name__ == '__main__':
    logger.basicConfig(
        level=logger.INFO,
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    db = DataBase()

    md = MovieData()
    md.read_movie_data()
    md.get_unique_values('genres')
    md.get_filtered_movie_data({
        'language': ['English'],
        'country': ['UK']
    })

    md.calculate_statistics(grouping=['country', 'title_year'])

    md.write_to_database(data_base=db)

    get_all_movie_data_query = f'''
        Select * from Movies
        '''

    get_all_genre_data_query = f'''
        Select * from Genres
        '''

    print(md.read_from_database(get_all_movie_data_query, data_base=db))

    print(md.read_from_database(get_all_genre_data_query, data_base=db))

    md.calculate_statistics(grouping=['country', 'title_year']).to_csv('Country_Year_Stats.csv')
    md.calculate_statistics().to_csv('Basic_Stats.csv')
    md.calculate_statistics(grouping=['director_name']).to_csv('Director_Level_Stats.csv')
    md.calculate_statistics(grouping='title_year').to_csv('Year_level.csv')

    print('Done')
