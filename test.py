#IMPORT LIBRARIES AND DATASET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from pathlib import Path

# Get the folder where this Python file is located
folder = Path(__file__).parent

# Import datasets
ratings_df = pd.read_csv(folder / 'ratings.csv')
movies_df = pd.read_csv(folder / 'movies.csv')

print(ratings_df.head())
print(movies_df.head())

#DATA PREPROCESSING

"""In movies dataset we have year along with the title"""
"""So first we will extract year from title and assign it to a new column"""

movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))', expand = True)
movies_df.head()

#Remove parantheses from year

movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand=True)
movies_df.head()

#Remove year from title

movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
movies_df.head()

#Remove all the whitespaces from title

movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

#Convert Genres into a list

movies_df['genres'] = movies_df.genres.str.split('|')
movies_df.head()

#One Hot Encoding of Genres

movies_copy = movies_df.copy()

for index, row in movies_df.iterrows():
  for genre in row['genres']:
    movies_copy.at[index, genre] = 1

movies_copy.head()

#Filling NAN values with 0 

movies_copy = movies_copy.fillna(0)
movies_copy.head()

#Now let's check ratings dataset

ratings_df.head()

#Timestamp column is not necessary, so we can drop it

ratings_df = ratings_df.drop(['timestamp'], axis=1)
ratings_df.head()

#CONTENT BASED RECOMMENDATION SYSTEM

"""Let's start by taking User Input for ratings of different movies"""

user_input = [
              {'title' : 'Grand Slam', 'rating' : 5.6},
              {'title' : 'Zero', 'rating' : 7},
              {'title' : 'Jumanji', 'rating' : 8.5},
              {'title' : 'Toy Story', 'rating' : 4.5}
]

movies_input = pd.DataFrame(user_input)
movies_input

#Add movieID to user input
#First we will filter selected movies from original dataset

input_id = movies_df[movies_df['title'].isin(movies_input['title'].tolist())]

#Merging the two datasets

movies_input = pd.merge(input_id, movies_input)
movies_input

#Drop the unnecessary columns like genres and year

movies_input = movies_input.drop(['genres','year'], axis=1)
movies_input

#Now we will check for same movies given in input in original dataset

movies_user = movies_copy[movies_copy['movieId'].isin(movies_input['movieId'].tolist())]
movies_user

#Reset index of this dataset 

movies_user = movies_user.reset_index(drop=True)
movies_user

#Create a Genre Table out of this dataset

UserGenreTable = movies_user.drop(['movieId','title','genres','year'], axis=1)
UserGenreTable

#dot product to get weights

UserProfile = UserGenreTable.transpose().dot(movies_input['rating'])

#User Profile for every genre

UserProfile

#Create a genre table for every movie in original datset

GenreTable = movies_copy.set_index(movies_copy['movieId'])
GenreTable

GenreTable = GenreTable.drop(['movieId','title','genres','year'], axis=1)
GenreTable.head()

#Final Recommendation value for each movie

Recommendation_df = ((GenreTable*UserProfile).sum(axis=1))/UserProfile.sum()
Recommendation_df.head()

#Sort the values to get movies with high recommendation values

Recommendation_df = Recommendation_df.sort_values(ascending=False)
Recommendation_df.head()

#Final recommendation table for movies

RecommendationTable =  movies_df.loc[movies_df['movieId'].isin(Recommendation_df.head(20).keys())]
RecommendationTable
