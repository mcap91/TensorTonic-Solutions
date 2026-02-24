def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x = x0
    
    # f(x)=ax^2+bx+c - the quadratic equation

    for _ in range(steps):
        gradient = 2 * a * x + b # the derivative f'(x)'
        x = x - lr * gradient  #this is the update step which multiplies by the learning rate
    
    return float(x)
    pass