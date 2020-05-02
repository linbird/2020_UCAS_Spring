import os
import sys

cwd = os.getcwd()
for person in os.listdir(cwd + "/data/photo/"):
    print(person)
    print("loading " + os.path.splitext(person)[0] + " Done")
