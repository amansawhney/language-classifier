
fileNames = ['english.txt', 'espanol.txt', 'norsk.txt']
keys = []
values = []
for fileName in fileNames:
    f = open(fileName, 'r')
    for line in f:
        keys.append(line.replace("\n", ""))
        values.append(fileName.replace(".txt", ""))
output_dict = dict(zip(keys, values))

download_dir = "lang_data.csv" #where you want the file to be downloaded to

csv = open(download_dir, "w")
#"w" indicates that you're writing strings to the file

columnTitleRow = "lang, word\n"
csv.write(columnTitleRow)

for key in output_dict.keys():
	lang = key
	word = output_dict[key]
	row = word + "," + " ".join(str(x) for x in list(lang)).replace(" ", ",") + "\n"
	csv.write(row)

