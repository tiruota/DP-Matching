import numpy as np
import math
import os
import sys

class Recognizer:
    def __init__(self, knowledge, diagonal_weight):
        self.knowledge = knowledge
        self.W = diagonal_weight
        self.matched_keys = []
        self.wdtable = []
        self.dp_planes = []
        self.cost_tables = []
        self.index_list = np.array(range(len(knowledge))).reshape([-1, 5, 3])

    def d_frame(self, frame_a, frame_b):
        mode = int(sys.argv[3])
        result = 0
        # power + Δpower
        if mode == 2:
            for i in range(2):
                result += pow((frame_a[i]-frame_b[i]), 2)
            return math.sqrt(result)
        # power
        elif mode == 1:
            result += pow((frame_a[0]-frame_b[0]), 2)
            return math.sqrt(result)
        # power + Δpower
        else:
            result += pow((frame_a[1]-frame_b[1]), 2)
            return math.sqrt(result)

    # creat graph
    def get_dp_plane(self, input_word, temp_word):
        dp_plane = []
        for frame_i in input_word:
            row_i = []
            for frame_j in temp_word:
                row_i.append(self.d_frame(frame_i, frame_j))
            dp_plane.append(row_i)
        self.dp_planes.append(dp_plane)
        return dp_plane

    # DP-matching
    def dp_matching(self, dp_plane):
        I = len(dp_plane) #I = # row
        J = len(dp_plane[0]) #J = length of row

        cost_table = [[0 for j in range(J)] for i in range(I)]
        
        #first row
        cost_table[0][0] = dp_plane[0][0]

        for j in range(J-1):
            cost_table[0][j+1] = cost_table[0][j] + dp_plane[0][j+1]

        #first column 
        for i in range(I-1):
            cost_table[i+1][0] = cost_table[i][0] + dp_plane[i+1][0]

        #rest of the table
        for i in range(I-1):
            for j in range(J-1): #current node is (i+1, j+1)                
                top = dp_plane[i+1][j+1] + cost_table[i+1][j]
                left = dp_plane[i+1][j+1] + cost_table[i][j+1]
                diag = self.W * dp_plane[i+1][j+1] + cost_table[i][j]
                cost_table[i+1][j+1] = min([top, left, diag])
        self.cost_tables.append(cost_table)
        return cost_table[I-1][J-1] / (I+J)

    def word_distance_table(self, test_data):
        table = []
        progress = 0

        for input_word in test_data:
            score_vector = []
            for temp_word in self.knowledge:
                score = self.dp_matching(self.get_dp_plane(input_word, temp_word))
                score_vector.append(score)
            table.append(score_vector)
            progress+=1
            print(str(progress) +"/" + str(len(test_data)))

        self.wdtable = table
        return table

    # calculate score
    def test(self, flist_test, test_data):
        table = self.word_distance_table(test_data)
        for score_vector in table:
            key = score_vector.index(min(score_vector))
            self.matched_keys.append(key)
        self.matched_keys = np.array(self.matched_keys).reshape([-1,5,3])

        match_list = []
        voice_list = []
        for i in range(len(self.matched_keys)):
            for j in range(len(self.matched_keys[0])):
                for k in range(len(self.matched_keys[0][0])):
                    key = self.matched_keys[i][j][k]
                    idx = (np.where(self.index_list == key))[1][0]
                    if idx == 0:
                        voice_list.append("いち")
                    elif idx == 1:
                        voice_list.append("に")
                    elif idx == 2:
                        voice_list.append("さん")
                    elif idx == 3:
                        voice_list.append("よん")
                    else:
                        voice_list.append("ご")
                    is_match = True if j == idx else False
                    match_list.append(is_match)
        
        for i in range(len(match_list)):
            print(flist_test[i], voice_list[i], match_list[i])

        print("test accuracy: %f"%(float(np.count_nonzero(match_list)/len(match_list))))
        
# print table (for debug)
def twodp(table):
    for row in table:
        print(row)

# read voice data
def read_folder(directory):
    file_list = []
    data_set = []
    entries_ = os.listdir(directory)
    entries = sorted(entries_)
    for entry in entries:
        f = open(directory+'/'+entry, 'r')
        word = f.read().split('\n')

        frames_of_word = []

        for frame in word:
            token = frame.split('\t')
            token = list(filter(lambda x: x != '', token))
            token = list(map(lambda x: float(x), token))
            if token:
                frames_of_word.append(token)
            
        file_list.append(entry)
        data_set.append(frames_of_word)
    
    return file_list, data_set

def main():
    flist_temp, template = read_folder(sys.argv[1])
    rec = Recognizer(template, 1)
    flist_test, test_set = read_folder(sys.argv[2])
    rec.test(flist_test, test_set)

if __name__ == '__main__':
    main()