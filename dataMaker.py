import numpy as np
import pandas as pd
import string

np.random.seed(10)

remove_n = 10
fileNames = [ 'espanol.txt', 'norsk.txt', 'english.txt']
keys = []
values = []
for fileName in fileNames:
    f = open(fileName, 'r')
    length = 0
    for line in f:
        length += 1
        if length > 60000:
            break
        printable = set(string.printable)
        filter(lambda x: x in printable, line)
        keys.append(line.replace("\n", ""))
        values.append(fileName.replace(".txt", ""))
output_dict = dict(zip(keys, values))

download_dir = "lang_data.csv" #where you want the file to be downloaded to

csv = open(download_dir, "w")
#"w" indicates that you're writing strings to the file
letter_header_string = ''
for i in range(1,50):
    letter_header_string += "L" + str(i) + ","
columnTitleRow = "lang, word," + letter_header_string + "\n"
csv.write(columnTitleRow)
for key in output_dict.keys():
    lang = key
    word = output_dict[key]
    row = word + "," + " ".join(str(x) for x in list(lang)).replace(" ", ",") + "\n"
    csv.write(row)


