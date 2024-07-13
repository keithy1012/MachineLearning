
class RegressionCostFunctions:
    def __init__(self):
        self.error = 0


    def MeanError(self, actual, predicted):
        error = 0
        if (len(actual) != len(predicted)):
            return False
        else:
            for i in range(len(actual)):
                error += (actual[i] - predicted[i])
        return error

    def MeanAbsoluteError(self, actual, predicted):
        error = 0
        if (len(actual) != len(predicted)):
            return False
        else:
            for i in range(len(actual)):
                error += abs(actual[i] - predicted[i])
        return error

    def MeanSquaredError(self, actual, predicted):
        error = 0
        if (len(actual) != len(predicted)):
            return False
        else:
            for i in range(len(actual)):
                error += pow(actual[i] - predicted[i], 2)
        return error