
# Checks the gradients by manually testing the rate of change
def check_derivative(x, y, dy):
    epsilon = 1e-4
    ngrad = y(x+epsilon) - y(x-epsilon)
    ngrad /= 2*epsilon

    return ngrad / dy(x)