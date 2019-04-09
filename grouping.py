import numpy as np
import pandas as pd
class Grouping():

    def __init__(self):
        self.groups = {}
    
    def addTweetsToGroups(self):
        data = pd.read_csv("postProcessedText.csv")
        for col in data.columns[1:]:
            self.groups[col] = []
            for _ , row in data.loc[data[col] == 1].iterrows():
               self.groups[col].append(row.Tweet)
        
if __name__ == "__main__":
    g = Grouping()
    g.addTweetsToGroups()