import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


class Bootstrap_kNN:
    """
    Bootstrap-k Nearest Neighbors. 
    """
    def __init__(self, k):
        self.NN = NearestNeighbors(n_neighbors=k, 
                              algorithm='ball_tree')
        self.kNN = KNeighborsClassifier(n_neighbors=k)
    
    def train(self, train, test, epochs):
        """
        """
        X_train = train[0]; y_train = train[1]
        X_test = train[0]; y_test = train[1]
        
        self.NN.fit(X_test)
        
        # Initialize with kNN
        self.kNN.fit(X_train, y_train)
        ystar_test = self.kNN.predict(X_test)
        
        for _ in range(epochs):
            for index, point in enumerate(X_test):
                # Find nearest neighbors
                _, inds = self.NN.kneighbors([point])
                
                # Obtain majority vote
                votes = []
                for ind in inds[0]:
                    votes.append(ystar_test[ind])
                                
                ystar_test[index] = max(set(votes), key = votes.count)
                
        print("Bootstrap-kNN Accuracy: ", np.sum(ystar_test==y_test)/len(y_test) * 100, "%")
        
        return ystar_test

# def alter_score(k_dist, k_score_differences):
#     """
#     Returns how much to alter score of current point.
#     Parameters:
#      - Proximity to the k samples
#      - Score differences of the k samples between current point
#     Score is altered proportional to score difference, and inversely 
#     proportional to proximity.
#     """
#     # k_dist[-1] is only a marker value
#     score_range = k_dist[-1] - k_dist[0]
#     new_dist = [score_range - (s - k_dist[0]) for s in k_dist]
    
#     if sum(new_dist) != 0:
#         # Weights are normalized to a value between 0~1
#         weights = [d/sum(new_dist) for d in new_dist]

#         # Calculate value shift
#         alter_value = sum(np.array(k_score_differences) * np.array(weights) * alpha)

#         return alter_value
#     return 0

# if __name__ == "__main__":
#     # Read dataset
#     print_load("Loading dataset...")
#     df = load_numerical_dataset(adjusted_dir + "adjusted_55.csv", "X")
#     vali = list(load_numerical_dataset(dataset_dir, "y")["y"])
#     reps = [np.array(rep) for rep in df["X"]]
    
#     print_complete("Loaded dataset.")
    
#     nearest_neighbors = NearestNeighbors(n_neighbors=k_samples+1, 
#                                         algorithm='ball_tree').fit(reps)

#     print_load("Starting training...")

#     for epoch in range(n_epochs):
#         # Generate KDTree at each epoch
#         for index, rep in enumerate(reps):
#             # Get k nearest data points
#             dist, ind = nearest_neighbors.kneighbors([rep])

#             # First item is the point itself, so we remove it
#             dist = dist[:,1:][0]; ind = ind[:,1:][0]
#             k_scores = [df["y"][i] - df["y"][index] for i in ind]
            
#             # Alter score 
#             df.iloc[index, df.columns.get_loc("y")] += alter_score(dist, k_scores)
#         if epoch % 50 == 0:
#             score_accuracy(list(df["y"]))
#     print_complete("Training done.")
