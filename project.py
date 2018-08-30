import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial

from sklearn.preprocessing import LabelEncoder
from collections import defaultdict



song_data =  pd.read_csv("song_data.csv")
song_data['song'] = song_data["title"].map(str) + " " + song_data["artist_name"]
del song_data['title']
del song_data['artist_name']

song_data.sort_values('user_id',ascending = True)
len(song_data['user_id'].unique())
song_data

def construct_cooccurence_matrix(user_songs, all_songs):
    print('Loading.......')
    ####################################
    #Get users for all songs in user_songs.
    ####################################
    user_songs_users = []        
    for i in range(0, len(user_songs)):
        item_data = song_data[song_data['song'] == user_songs[i]]
        user_songs_users.append(item_data['user_id'].unique())
#         user_songs_users.append(item_users)
#     print(user_songs_users)
    ###############################################
    #Initialize the item cooccurence matrix of size 
    #len(user_songs) X len(songs)
    ###############################################
    cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)

    #############################################################
    #Calculate similarity between user songs and all unique songs
    #in the training data
    #############################################################
    for i in range(0,len(all_songs)):
        #Calculate unique listeners (users) of song (item) i
        songs_i_data = song_data[song_data['song'] == all_songs[i]]
        users_i = set(songs_i_data['user_id'].unique())

        for j in range(0,len(user_songs)):       

            #Get unique listeners (users) of song (item) j
            users_j = user_songs_users[j]

            #Calculate intersection of listeners of songs i and j
            users_intersection = users_i.intersection(users_j)

            #Calculate cooccurence_matrix[i,j] as Jaccard Index
            if len(users_intersection) != 0:
                #Calculate union of listeners of songs i and j
                users_union = users_i.union(users_j)

                cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union))
            else:
                cooccurence_matrix[j,i] = 0

    return cooccurence_matrix


d = {}
for i,s in enumerate(song_data['song']): 
     d[i] = s 
a = int(input("Press the column number you want to have recomendation for..."))


user_songs = [d[a]]
print(user_songs)
all_songs = song_data['song'].unique() 
cooccurence_matrix = construct_cooccurence_matrix(user_songs,all_songs)


user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
user_sim_scores = np.array(user_sim_scores)[0].tolist()

def top_recommendations(user_sim_scores,all_songs,user_songs):
    print('Recommendation For You')
    sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)

    #Create a dataframe from the following
    columns = ['song', 'score', 'rank']
    #index = np.arange(1) # array of numbers for the number of samples
    df = pd.DataFrame(columns=columns)
    user = song_data['user_id']
    #Fill the dataframe with top 10 item based recommendations
    rank = 1 
    for i in range(0,len(sort_index)):
        if ~np.isnan(sort_index[i][0]) and sort_index[i][0] !=0 and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
            df.loc[len(df)]=[all_songs[sort_index[i][1]],sort_index[i][0],rank]
            rank = rank+1
    if df.shape[0] == 0:
        print("The current user has no songs for training the item similarity based recommendation model.")
        return -1
    else:
        return df
    
df = top_recommendations(user_sim_scores,all_songs,user_songs)
print(df)
