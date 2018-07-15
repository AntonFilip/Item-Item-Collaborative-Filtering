import time
from math import sqrt
from random import randint

import pandas as pd
import numpy as np
import sys
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def loadUserRatingsForSong():
    for (index, user, song, rating) in songsDataset.itertuples():
        if (song in userRatingsForSongDict):
            usersWhoRatedCurrentSongDict = userRatingsForSongDict[song]
            usersWhoRatedCurrentSongDict[user] = rating
        else:
            userRatingsForSongDict[song] = {user: rating}

def calcSimilarSongs():
    for currentSong in uniqueSongsList:
        if currentSong == targetSong:
            continue
        if targetUser not in userRatingsForSongDict[currentSong]:
            continue
        nominator = 0.0
        denominatorSumTarget = 0.0
        denominatorSumCurrent = 0.0
        for user in targetSongUserRatingsDict:
            if user == targetUser:
                continue
            if user in userRatingsForSongDict[currentSong]:
                nominatorTargetSong = targetSongUserRatingsDict[user] - targetMeanRating #meanRatingsUsers.at[user, 'rating_mean'] za adjusted cosine
                nominatorCurrentSong = userRatingsForSongDict[currentSong][user] - meanRatings.at[currentSong, 'rating_mean'] #meanRatingsUsers.at[user, 'rating_mean'] za adjusted cosine
                nominator += nominatorTargetSong * nominatorCurrentSong
                denominatorSumTarget += nominatorTargetSong ** 2
                denominatorSumCurrent += nominatorCurrentSong ** 2
        denominator = sqrt(denominatorSumTarget * denominatorSumCurrent)
        if denominator == 0 or nominator < 0: # or nominator < 0 or (nominator / denominator) >= 0.9999
            continue
        else:
            similarityForSongDict[currentSong] = nominator / denominator
            #print("Similar song is ", currentSong, " ", songCorrs[currentSong])

def calcFinalScore():
    if not len(similarityForSongDict):
        print("There are no similar songs!")
        return 0
    i = 0
    similarityRatingSum = 0
    similaritySum = 0
    for (song, corr) in sorted(similarityForSongDict.items(), key=lambda x: x[1], reverse=True):
        #if i == 80:
        #    break
        usersWhoRatedCurrentSongDict = userRatingsForSongDict[song]
        if targetUser not in usersWhoRatedCurrentSongDict:
            continue
        #    userRatingsForSong[song][targetUser] = meanRatings.at[song, 'rating_mean']
        similarityRatingSum += usersWhoRatedCurrentSongDict[targetUser] * similarityForSongDict[song]
        similaritySum += abs(similarityForSongDict[song])
        i += 1
    weightedScore = similarityRatingSum / similaritySum
    return weightedScore


sns.set_style('white')
#load data
songsDataset = pd.read_csv('songsDataset.csv')
#calculate mean rating for every song
meanRatings = pd.DataFrame(songsDataset.groupby('songID')['rating'].mean())
meanRatings.rename(columns = {'rating': 'rating_mean'}, inplace=True)
#calculate mean rating for every user
meanRatingsUsers = pd.DataFrame(songsDataset.groupby('userID')['rating'].mean())
meanRatingsUsers.rename(columns = {'rating': 'rating_mean'}, inplace=True)
uniqueSongsList = songsDataset.songID.unique()
userRatingsForSongDict = {}
loadUserRatingsForSong()
similarityForSongDict = {}

#targetUser = 57163
#targetSong = 132189

rmseSum = 0
maeSum = 0
targetSongs = []

for x in range(100):
    targetSongs.append(uniqueSongsList[randint(0, len(uniqueSongsList) - 1)])

k = 0
for song in targetSongs:
    targetSong = song
    targetMeanRating = meanRatings.at[targetSong, 'rating_mean']
    print("Mean rating for song is: ", targetMeanRating)
    predictedScores = []
    actualScores = []
    j = 0
    for targetUser in userRatingsForSongDict[targetSong]:
        if (j == 500):
            break
        j += 1
        similarityForSongDict = {}

        start = time.time()
        targetSongUserRatingsDict = userRatingsForSongDict[targetSong]
        actualTargetRating = userRatingsForSongDict[targetSong][targetUser]

        calcSimilarSongs()

        finalScore = calcFinalScore()

        print(targetUser, finalScore, actualTargetRating)
        print("")
        if finalScore != 0:
            predictedScores.append(finalScore)
            actualScores.append(actualTargetRating)

    if (actualScores and predictedScores):
        rmse = sqrt(mean_squared_error(actualScores, predictedScores))
        rmseSum += rmse
        mae = mean_absolute_error(actualScores, predictedScores)
        maeSum += mae
        print("Root mean squared error is: ", rmse, "and Mean absolute error is: ", mae)
        k += 1

print("Mean rmse is: ", rmseSum/k, "and Mean absolute error is: ", maeSum/k)