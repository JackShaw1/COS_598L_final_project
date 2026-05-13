import csv

ctr = {}

with open("scan_proteome/imatinib_pocket_hits.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        ctr[row[0]] = ""

print(len(ctr))