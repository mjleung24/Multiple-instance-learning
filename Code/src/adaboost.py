import numpy as np
from scipy.spatial import distance

class AdaBoostMIL:
    """
    Construct a new with input training data and provided neural network
    """
    def __init__(self, X, Y, distance_param = 'euclid', max_hypothesis = 0):
        # weights of the weak hypothesis (normalized)
        self.hWeights = []
        # hypothesis stores (positive bag, instance)
        self.hypotheses = []
        # bag weights
        self.bWeights = np.full(len(Y), 1/len(Y))
        # stores bag/instance pair's distance to other bags
        self.radii = {}
        # stores optimal radius using (bag, instance) as key
        self.optimal_radius = {}
        self.param = distance_param
        self.Y = Y
        self.X = X
        # limit the number of hypotheses (balls)
        self.max_hypothesis = max_hypothesis



    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        def find_shortest_distance(center, points):
            center = np.array(center)
            points = np.array(points)
            if self.param == 'euclid':
                d = np.sqrt(np.sum((points - center)**2, axis = 1))
            elif self.param == 'manhattan':
                d = [distance.cityblock(center, point) for point in points]
            elif self.param == 'chebyshev':
                d = [distance.chebyshev(center, point) for point in points]
            elif self.param == 'cosine':
                d = [1-distance.cosine(center, point) for point in points]
            elif self.param == 'minkowski':
                d = [distance.minkowski(center, point, p=5) for point in points]
            else:
                d = [distance.euclidean(center, point) for point in points]
            return np.min(d)
        
        #num_pos_bags = len(Y[Y==1])
        # bag i
        for i in range(len(X)):
            # instance j
            num_hypothesis_per_bag = 0
            for j in range(len(X[i])):
                # stores distance from center x_j to bag_k
                bagDistance = []
                if Y[i] == 1:
                    # increment the number of hypothesis by 1
                    num_hypothesis_per_bag += 1

                    # stop adding hypothesis once the number of hypotheses in that bag exceed the limit per bag
                    #if self.max_hypothesis != 0 and num_hypothesis_per_bag > (self.max_hypothesis / num_pos_bags):
                        #continue
                    # looping through other bags
                    for k in range(len(X)):
                        if k==i:
                            k+=1
                        if k<len(X):
                            bagDistance.append([k, find_shortest_distance(X[i][j], X[k])])
                    # stores [positive bag, instance]
                    self.hypotheses.append((i,j))
                    self.radii[(i, j)] = bagDistance
        
        # looping through all instances in positive bags as centers
        for hypothesis in range(len(self.hypotheses)):

            sorted_array = sorted(self.radii[self.hypotheses[hypothesis]], key=lambda x: x[1])
            # center point bag[i] instance[j]
            center = X[self.hypotheses[hypothesis][0]][self.hypotheses[hypothesis][1]]
            best_bag_index = -1
            best_radius = 0
            best_acc = self.bWeights[self.hypotheses[hypothesis][0]]
            # find optimal radius
            #sorted_bags = np.array(sorted_array)[:,0]
            sorted_bags = [row[0] for row in sorted_array]
            for bag_index in range(len(sorted_bags)):
                # all bags within the current radius are predicted positive
                pos_bags = sorted_bags[:bag_index+1]
                # all bags outside the current radius are predicted negative
                neg_bags = sorted_bags[bag_index+1:]

                # trivially predicts center bag as positive
                weighted_acc = self.bWeights[self.hypotheses[hypothesis][0]]
                
                # calculates the distribution accuracy: sum of the weights of the bags classified correctly
                pos_correct = [self.bWeights[index] for index in pos_bags if Y[index] == 1]
                neg_correct = [self.bWeights[index] for index in neg_bags if Y[index] != 1]
                weighted_acc += (sum(pos_correct)+sum(neg_correct))

                # if the distribution accuracy of the radius chosen is greater than the previous optimal radius, then it will be updated
                if weighted_acc > best_acc:
                    best_acc = weighted_acc
                    best_bag_index = bag_index
                    best_radius = sorted_array[bag_index][1]
            self.optimal_radius[self.hypotheses[hypothesis]] = best_radius

            # update error/weights/alpha
            err = self.compute_error(sorted_bags[:best_bag_index+1], sorted_bags[best_bag_index+1:], Y)
            alpha = self.compute_alpha(err)
            self.hWeights.append(alpha)
            for i in range(len(self.bWeights)):
                # No point in adjusting the weight of current bag because it will be classified as positive
                if i != self.hypotheses[hypothesis]:
                    true_label = Y[i]
                    if i in sorted_bags[:best_bag_index+1]: pred_label = 1
                    else: pred_label = -1
                    # only updates the weight if prediction is different from true label
                    if true_label != pred_label:
                        self.bWeights[i] = self.update_weights(self.bWeights[i], alpha, 1)
                        self.bWeights = self.bWeights/sum(self.bWeights)

                
    def boost_predict(self, X):
        final_predict = np.zeros(len(X))
        for i in range(len(self.hypotheses)):
            h_predict = self.hypothesis_predict(i, X)
            final_predict += (h_predict* self.hWeights[i])
        return np.sign(final_predict)

    def hypothesis_predict(self, hypothesis, X):
        return np.array([self.__predict_single(hypothesis, bag) for bag in X])

    def __predict_single(self, hypothesis, X):
        center = self.X[self.hypotheses[hypothesis][0]][self.hypotheses[hypothesis][1]]
        radius = self.optimal_radius[self.hypotheses[hypothesis]]
        return self.find_positive(center, radius, X)

    def find_positive(self, center, radius, points):
        distances = self.distance_function(center, points)
        if np.any(distances <= radius): return 1
        return -1

    def distance_function(self, center, points):
        if self.param == 'euclid': return np.linalg.norm(points-center, axis=1)
        elif self.param == 'manhattan':
            d = [distance.cityblock(center, point) for point in points]
        elif self.param == 'chebyshev':
            d = [distance.chebyshev(center, point) for point in points]
        elif self.param == 'cosine':
            d = [1-distance.cosine(center, point) for point in points]
        elif self.param == 'minkowski':
            d = [distance.minkowski(center, point, p=5) for point in points]
        else:
            d = [distance.euclidean(center, point) for point in points]
        return np.linalg.norm(points-center, axis=1)

    # Compute error rate, alpha and w
    def compute_error(self, pos_bags, neg_bags, Y):
        err = [self.bWeights[index] for index in pos_bags if Y[index] != 1]
        err += [self.bWeights[index] for index in neg_bags if Y[index] == 1]
        return (sum(err)/sum(self.bWeights))

    def compute_alpha(self, error):
        if error == 0: error = 0.00000001
        return np.log((1 - error) / error)

    def update_weights(self, w_i, alpha, err):
        return w_i * np.exp(alpha * err)
    
    


