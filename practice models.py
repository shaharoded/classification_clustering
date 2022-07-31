import math
#create class
class SurfBoards:
    def __init__(self,length,width,thickness,category):
        self.l = length * 33 #feet * 33 (cm)
        self.w = width * 2.5 #inch * 2.5 (cm)
        self.t = thickness * 2.5 #inch * 2.5 (cm)
        self.type = category #in ["shortboard","fishboard","longboard","midrange","boogyboard"]
    def get_measures(self):
        return [self.l,self.w,self.t]
    def get_type(self):
        return self.type
    def __repr__(self):
        return f"{self.type} size {self.l/33}ft"

b18 = SurfBoards(5.7,24,2.3,"fishboard")
b2 = SurfBoards(5.3,26,2.4,"fishboard")
b3 = SurfBoards(5.9,19,2.2,"shortboard")
b4 = SurfBoards(6.0,21,2,"shortboard")
b5 = SurfBoards(6.1,20,2.1,"shortboard")
b6 = SurfBoards(6.7,19,2.3,"shortboard")
b17 = SurfBoards(6.5,22,2.2,"midrange")
b8 = SurfBoards(5.6,20,2,"shortboard")
b11 = SurfBoards(7,21,2.5,"midrange")
b20 = SurfBoards(6.2,21,2,"midrange")
b9 = SurfBoards(5.0,28,2.1,"fishboard")
b12 = SurfBoards(7.1,21,2,"midrange")
b13 = SurfBoards(7.2,20,2,"midrange")
b14 = SurfBoards(8.0,19,2,"longboard")
b15 = SurfBoards(7.8,18.2,2.2,"longboard")
b16 = SurfBoards(9.0,19,2,"longboard")
b7 = SurfBoards(8.5,20,2,"longboard")
b1 = SurfBoards(3.5,28,3,"boogyboard")
b19 = SurfBoards(3.8,30,3,"boogyboard")
b10 = SurfBoards(3.0,35,3,"boogyboard")
test_quiver = [b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20]
#create K Means Model
def distance_func_for_KNN(surfboardA,surfboardB):
    #volumeA = surfboardA.get_measures()[0]*surfboardA.get_measures()[1]*surfboardA.get_measures()[2]
    #volumeB = surfboardB.get_measures()[0]*surfboardB.get_measures()[1]*surfboardB.get_measures()[2]
    dist = float(((surfboardA.get_measures()[0] - surfboardB.get_measures()[0])**2) +
                 ((surfboardA.get_measures()[1] - surfboardB.get_measures()[1])**2)+
                 ((surfboardA.get_measures()[2] - surfboardB.get_measures()[2])**2))
    dist = math.sqrt(dist)
    if dist == 0:
        return math.inf
    else:
        return abs(round(dist,2))

def distance_func_for_K_Means(surfboardA,surfboardB):
    #volumeA = surfboardA.get_measures()[0]*surfboardA.get_measures()[1]*surfboardA.get_measures()[2]
    #volumeB = surfboardB.get_measures()[0]*surfboardB.get_measures()[1]*surfboardB.get_measures()[2]
    dist = float(((surfboardA.get_measures()[0] - surfboardB.get_measures()[0])**2) +
                 ((surfboardA.get_measures()[1] - surfboardB.get_measures()[1])**2)+
                 ((surfboardA.get_measures()[2] - surfboardB.get_measures()[2])**2))
    dist = math.sqrt(dist)
    return abs(round(dist,2))
import random
def K_Means_model(quiver,dist_func,n_centers,thresh):
    random.shuffle(quiver)
    keys = [i for i in range(n_centers)]
    centers = [quiver[key] for key in keys] #choosing n initial centers
    memo = {}
    for i in centers:
        memo[i] = [] #set random boards as centers
    #assighn quiver to closest center
    def find_closest_center(board,centers):
        dist_list = [dist_func(board,center) for center in centers]
        closest_center = centers[dist_list.index(min(dist_list))]
        return closest_center
    for board in quiver:
        memo[find_closest_center(board,centers)].append(board)
    #second part - moving the centers within the groups
    def centers_adjustment(memo):
        new_memo = {}
        for key in memo.keys():
            boards = memo[key]
            distances = []
            for board in boards:
                sum_of_distances = sum([dist_func(board,i) for i in boards])
                distances.append(sum_of_distances)
            new_center = boards[distances.index(min(distances))]
            new_memo[new_center] = []
        new_centers = list(new_memo.keys())
        for i in quiver:
            new_memo[find_closest_center(i, new_centers)].append(i)
        return new_memo
    while thresh >= 0:
        memo = centers_adjustment(memo)
        thresh = thresh - 1
    return memo

print("K-Means clusters:")
print(K_Means_model(test_quiver,distance_func_for_K_Means,3,900))
print("-"
      "-"
      "-"
      "-"
      "-")


#create KNN Model
def KNN_model(quiver,neighbors,dist_func):
    def nearest_neighbors(board):
        distances = [dist_func(board,i) for i in quiver] #list of distances from each board in quiver
        n_indexes = []
        for i in range(neighbors):
            curr_min = min(distances)
            n_indexes.append(distances.index(curr_min))
            distances.pop(distances.index(curr_min))
        return [quiver[i] for i in n_indexes] #return list of nearest neighbors from quiver
    def predicted_category(board):
        neighbors = nearest_neighbors(board)
        tmp = [i.get_type() for i in neighbors]
        categories_dict = {}
        for i in tmp: #find which category is most popular
            if i in categories_dict.keys():
                categories_dict[i] = categories_dict[i] + 1
            else:
                categories_dict[i] = 1
        max_val = max(categories_dict.values())
        chosen_category = 0
        for key in categories_dict.keys():
            if categories_dict[key] == max_val:
                chosen_category = key
                break
        return str(board),board.get_type(),chosen_category #returns tuple
    return [predicted_category(i) for i in quiver]
print("KNN classification prediction:")
print(KNN_model(test_quiver,3,distance_func_for_KNN))
#create confusion matrix
import pandas as pd
results = KNN_model(test_quiver,3,distance_func_for_KNN)

def confusion_matrix(model_results):
    prediction = [(i[1],i[2]) for i in model_results]
    tmp = []
    for i in prediction:
        tmp.append(i[0])
        tmp.append(i[1])
    categories = set(tmp)
    matrix = [[0 for i in categories] for i in categories]
    index_dict = {}
    counter = 0
    for i in categories:
        index_dict[i] = counter
        counter += 1
    #load real values into matrix:
    for key in index_dict.keys():
        for i in prediction:
            if i[0] == key:
                prediction_index = list(index_dict.keys()).index(i[1])
                matrix[index_dict[key]][prediction_index] += 1
    #create full matrix
    print(prediction)
    for i in range(len(matrix)):
        matrix[i] = [list(index_dict.keys())[i]] + matrix[i]
    heads = ["x"] + list(index_dict.keys())
    full_matrix = [heads] + matrix
    df = pd.DataFrame(full_matrix)
    return df
print(confusion_matrix(results))
