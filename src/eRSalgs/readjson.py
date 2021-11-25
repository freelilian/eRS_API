# read signature data from json 
import json
import csv
import numpy as np

with open('../raw_data_v3/movies.json') as json_file:
    data = json.load(json_file)

data_file = open('../data_v3/signature_v3.csv', 'w')
csv_writer = csv.writer(data_file)
count = 0

for movie in data:
    if count == 0:
        top_level_header = movie.keys()
        id_header = list(top_level_header)[1]
        count += 1
    signature = movie['signature']
    
    if count == 1:
        signature_header = list(signature.keys())    
        signature_header.insert(0, id_header)
        count += 1
        csv_writer.writerow(signature_header)
            # titleId	anger	anticipation	disgust	fear	joy	negative	positive	sadness	surprise	trust
    
    id_values = list(movie.values())[1] 
        # should be a string
        # starting from 4337th entry, titleId is a string starting with 'tt'
        # 19753 movies
    if id_values[:2] == 'tt':
        id = int(id_values[2:])
    else:
        id = int(id_values)
        
    signature_entry = list(signature.values())
    signature_entry.insert(0, id)
    
    csv_writer.writerow(signature_entry)
    
data_file.close()

