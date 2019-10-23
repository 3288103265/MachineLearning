import numpy as np

thresh = 1e-5

# Prepare data
x = np.array([[0.697, 0.46, 1.],
              [0.774, 0.376, 1.],
              [0.634, 0.264, 1.],
              [0.608, 0.318, 1.],
              [0.556, 0.215, 1.],
              [0.403, 0.237, 1.],
              [0.481, 0.149, 1.],
              [0.437, 0.211, 1.],
              [0.666, 0.091, 1.],
              [0.243, 0.267, 1.],
              [0.245, 0.057, 1.],
              [0.343, 0.099, 1.],
              [0.639, 0.161, 1.],
              [0.657, 0.198, 1.],
              [0.36, 0.37, 1.],
              [0.593, 0.042, 1.],
              [0.719, 0.103, 1.]])

y = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
y = y.transpose()

# Define Objective function.
def loss(x, y, b):
    sum = 0

    for xi, yi in zip(x, y):
        xi = xi.reshape(xi.shape[0], 1)
        b_T_x = np.dot(b.T, xi)
        sum += -yi * b_T_x + np.log(1 + np.exp(b_T_x))

    return sum

# Train.
def run(x, y, iterate=100):
    beta = np.zeros((x.shape[1], 1))
    beta[-1] = 1
    old_l = 0
    cur_iter = 0

    while cur_iter < iterate:
        cur_iter += 1
        cur_l = loss(x, y, beta)
        if np.abs(cur_l - old_l) <= thresh:
            break

        old_l = cur_l
        d1_beta, d2_beta = 0, 0
        # Update parameter. Get from https://blog.csdn.net/weixin_37922777/article/details/88625728
        for xi, yi in zip(x, y):
            xi = xi.reshape(xi.shape[0], 1)
            p1 = p1_function(xi, beta)

            d1_beta -= np.dot(xi, yi - p1)
            d2_beta += np.dot(xi, xi.T) * p1 * (1 - p1)

        try:
            beta = beta - np.dot(np.linalg.inv(d2_beta), d1_beta)
        except Exception as e:
            break
    return beta


def p0_function(xi, beta):
    return 1 - p1_function(xi, beta)


def p1_function(xi, beta):
    beta_T_x = np.dot(beta.T, xi)
    exp_beta_T_x = np.exp(beta_T_x)

    return exp_beta_T_x / (1 + exp_beta_T_x)


b = run(x, y)
accuracy = 0
for xi, yi in zip(x, y):
    p1 = p1_function(xi, b)
    judge = 0 if p1 < 0.5 else 1
    accuracy += (judge == yi[0])
error = 1 - accuracy / x.shape[0]
print('Error:', error * 100, '%')
