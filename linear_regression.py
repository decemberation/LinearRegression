def gradient(x, y, theta):
    dJ0 = 0.0
    dJ1 = 0.0
    J = 0
    m = len(y)
    for i in range(0, m):
        h_theta = theta[0] + theta[1]*x[i]
        dJ0 = dJ0 + h_theta - y[i]
        dJ1 = dJ1 + (h_theta - y[i])*x[i]
        J = J + (h_theta - y[i])**2
    dJ0 = dJ0/m
    dJ1 = dJ1/m
    J = J/(2*m)
    return (J, dJ0, dJ1)

def gradient_descent(x, y):
    iters = 5000
    theta = [0, 0]
    alpha = 0.00001
    for i in range(0, iters):
        J, dJ0, dJ1 = gradient(x, y, theta)
        print("Iter %d: %.3f" % (i+1, J))
        theta[0] = theta[0] - alpha*dJ0
        theta[1] = theta[1] - alpha*dJ1
    return theta

if __name__ == '__main__':
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y = [0.5, 2.1, 3.9, 5.5, 9, 10.3, 12, 14, 15, 21]
    theta = gradient_descent(x, y)
    print("Model: h(x) = %.3f + %.3fx" % (theta[0], theta[1]))
    for i in range(0, len(x)):
        hx = theta[0] + theta[1]*x[i]
        print("x: %d, y: %.3f, h(x): %.3f" % (x[i], y[i], hx))
