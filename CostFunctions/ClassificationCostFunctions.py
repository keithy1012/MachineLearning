import math
class RegressionCostFunctions:
    def __init__(self):
        self.error = 0
    
    def BinaryCrossEntropy(prob_value, actual_value):
        return - (actual_value * math.log(prob_value) + (1-actual_value)*math.log(1-prob_value))
    
    def CrossEntropy(prob_dist, actual_dist):
        return -1 * sum(x_i*math.log(y_i) for x_i, y_i in zip(prob_dist, actual_dist))