import pandas as pd

song_data =  pd.read_csv("songs.csv")
d = {}
for i,s in enumerate(song_data['song']): 
     d[i] = s 
df = pd.DataFrame.from_dict(d,orient = 'index')     
print(df)   
