fileNames = ['english.txt', 'espanol.txt', 'norsk.txt']
keys = []
values = []
for fileName in fileNames:
    f = open(fileName, 'r')
    for line in f:
        keys.append(line.replace("\n", ""))
        values.append(fileName.replace(".txt", ""))
output_dict = dict(zip(keys, values))
print output_dict



