import os
import sys
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import pyrcc
from uis import res
from PyQt5 import QtCore, QtGui, QtWidgets
from functools import partial
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QTimer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from uszipcode import SearchEngine
from sklearn.cluster import KMeans
import nltk
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi("uis\GUI.ui",self)
        self.GUI_initialize_properities()
        self.GUI_initialize_Objects()
        self.GUI_initialize_Buttons()
        self.prepare_Cosine_similarities()
        self.auto_recommend()
    def GUI_initialize_properities(self):
        self.setWindowIcon(QIcon("uis\materials\systemLogo.png"))
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
    def GUI_initialize_Objects(self):
        self.movieTitle_tbox = self.findChild(QLineEdit, "movieTitle_tbox")
        self.movieCategory_tbox = self.findChild(QLineEdit, "movieCategory_tbox")
        self.top10_title_label = self.findChild(QLabel, "top10_title_label")
        self.recomendedMovies_listWidget = self.findChild(QListWidget, "recomendedMovies_listWidget")

    def GUI_initialize_Buttons(self):
        # Getting process signal button
        self.recommend_btn = self.findChild(QPushButton, "recommend_btn")
        self.recommend_btn.clicked.connect(self.recommend_10_Movies)

        self.search_btn = self.findChild(QPushButton, "search_btn")
        self.search_btn.clicked.connect(self.seachByCategory)

        self.randomMovieName_btn = self.findChild(QPushButton, "randomMovieName_btn")
        self.randomMovieName_btn.clicked.connect(self.mockTest_pick_movieTitle)


        #------ Exit Button ------
        self.exit_btn = self.findChild(QPushButton, "exit_btn")
        self.exit_btn.clicked.connect(self.close)
    def auto_recommend(self):
        # Sort the DataFrame by 'release_year' in descending order
        sorted_movies = self.movies_data.sort_values('release_year', ascending=False)
        # Select the top 10 rows (newest movies)
        top_10_newest_movies = sorted_movies.head(20)
        for movie in top_10_newest_movies['title']:
            movie_info = movie + " - " + self.movies_data.loc[self.movies_data['title'] == movie, 'genres'].values[0]
            self.recomendedMovies_listWidget.addItem(movie_info)
            self.recomendedMovies_listWidget.addItem(" ")

    def recommend_10_Movies(self):
        self.recomendedMovies_listWidget.clear()
        if self.movieTitle_tbox.text() in self.movies_data['title'].values:
            movie_Name = self.movieTitle_tbox.text()
            # ------------------------------------------------------
            selected_movie = self.movies_data.loc[self.movies_data['title'] == movie_Name]
            idx = self.movies_data.loc[self.movies_data['title'] == movie_Name].index[0]
            release_year_diff = []
            similarity_scores = list(enumerate(self.cosine_similarities[idx]))
            target_year = selected_movie['release_year']
            self.movies_data['release_year_diff'] = abs(self.movies_data['release_year'] - target_year)
            similarity_scores = sorted(similarity_scores,key=lambda x: (x[1], -self.movies_data.iloc[x[0]]['release_year_diff']), reverse=True)
            top_10_similar_movies = [(self.movies_data.iloc[similarity_score[0]]['title'], similarity_score[1]) for
                                     similarity_score in similarity_scores if similarity_score[0] != idx][:10]
            # print("Top 10 similar Movies for '{}':".format(movie_Name))
            self.top10_title_label.setText("Top 10 similar Movies for '{}' :".format(movie_Name))
            for movie, similar_movie_scores in top_10_similar_movies:
                Recomended_Movie_Title = movie
                Recomended_Movie_similarity_with_user_movie = round(similar_movie_scores, 4)
                # print("- '{}' with similarity score {}".format(Recomended_Movie_Title,Recomended_Movie_similarity_with_user_movie))
                movie_info = Recomended_Movie_Title +" - "+ self.movies_data.loc[self.movies_data['title'] == Recomended_Movie_Title, 'genres'].values[0]
                self.recomendedMovies_listWidget.addItem(movie_info)
                self.recomendedMovies_listWidget.addItem(" ")

    def seachByCategory(self):
        # Filter the DataFrame based on the genre
        self.top10_title_label.setText("Recommended "+self.movieCategory_tbox.text().title()+" Movies")

        filtered_movies = self.movies_data[self.movies_data['genres'] == self.movieCategory_tbox.text().title()]
        if not filtered_movies.empty:
            self.movieCategory_tbox.setText("")
            sorted_movies = filtered_movies.sort_values('release_year', ascending=False)
            self.recomendedMovies_listWidget.clear()
            for movie in sorted_movies['title']:
                movie_info = movie + " - " + self.movies_data.loc[self.movies_data['title'] == movie, 'genres'].values[
                    0]
                self.recomendedMovies_listWidget.addItem(movie_info)
                self.recomendedMovies_listWidget.addItem(" ")

    def mockTest_pick_movieTitle(self):
        self.movieTitle_tbox.setText(self.movies_data['title'].sample().values[0])

    def prepare_Cosine_similarities(self):
        self.movies_data = pd.read_csv('movies_modif.csv')
        self.movies_data = self.movies_data.drop('Unnamed: 3', axis=1)
        # ------------------------------------------------------------------------------
        self.movies_data['genres'] = self.movies_data['genres'].str.replace("|", " ")
        self.movies_data['genres'] = self.movies_data['genres'].str.replace("'", "")
        # Apply TF-IDF on movie genres
        vectorizer = TfidfVectorizer()
        # Fit and transform the genres records
        tfidf_matrix = vectorizer.fit_transform(self.movies_data['genres'])
        # Calculate cosine similarity matrix
        self.cosine_similarities = cosine_similarity(tfidf_matrix)
        # Extract movie release year from title to be used in sorting
        self.movies_data['release_year'] = [self.extract_release_year(movie) for movie in self.movies_data['title']]

    # Function to extract release year from movie title using regix
    def extract_release_year(self, title):
        pattern = r"\((\d{4})\)$"
        match = re.search(pattern, title)
        if match:
            # print(match.group(1))
            return int(match.group(1))
        else:
            return None

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window =Window()
    window.show()
    sys.exit(app.exec())