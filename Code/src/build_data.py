import numpy as np

def linear(slope, intercept, x):
    return slope * x + intercept

def quadratic(a, b, c, x):
    return a * (x ** 2) + linear(b, c, x)

def cubic(a, b, c, d, x):
    return a * (x ** 3) + quadratic(b, c, d, x)

def generate_linear_data(bag_count = 100, instances_per_bag = 10, 
                        params = [1, 0],
                        min_x = -10,
                        max_x = 10):
    data = []
    labels = []
    for _ in range(bag_count):
        data.append(np.array([np.array([x := np.random.random() * (max_x - min_x) + min_x, linear(params[0], params[1], x)]) for _ in range(instances_per_bag)]))
        labels.append(1)
    return np.array(data), np.array(labels)

def generate_quadratic_data(bag_count = 100, instances_per_bag = 10,
                            params = [1, 0, 0],
                            min_x = -10,
                            max_x = 10):
    data = []
    labels = []
    for _ in range(bag_count):
        data.append(np.array([np.array([x := np.random.random() * (max_x - min_x) + min_x, quadratic(params[0], params[1], params[2], x)]) for _ in range(instances_per_bag)]))
        labels.append(2)
    return np.array(data), np.array(labels)



def generate_cubic_data(bag_count = 100, instances_per_bag = 10,
                            params = [1, 0, 0, 0],
                            min_x = -10,
                            max_x = 10):
    data = []
    labels = []
    for _ in range(bag_count):
        data.append(np.array([np.array([x := np.random.random() * (max_x - min_x) + min_x, cubic(params[0], params[1], params[2], params[3], x)]) for _ in range(instances_per_bag)]))
        labels.append(3)
    return np.array(data), np.array(labels)

def generate_data(bags_per_function = 100, instances_per_bag = 10, 
              linear_params = [1, 0],
              quadratic_params = [1, 0, 0],
              cubic_params = [1, 0, 0, 0],
              min_x = -10,
              max_x = 10):
    data = []
    labels = []
    for _ in range(bags_per_function):
        data.append(np.array([np.array([x := np.random.random() * (max_x - min_x) + min_x, linear(linear_params[0], linear_params[1], x)]) for _ in range(instances_per_bag)]))
        labels.append(1)


    for _ in range(bags_per_function):
        data.append(np.array([np.array([x := np.random.random() * (max_x - min_x) + min_x, quadratic(quadratic_params[0], quadratic_params[1], quadratic_params[2], x)]) for _ in range(instances_per_bag)]))
        labels.append(2)


    for _ in range(bags_per_function):
        data.append(np.array([np.array([x := np.random.random() * (max_x - min_x) + min_x, cubic(cubic_params[0], cubic_params[1], cubic_params[2], cubic_params[3], x)]) for _ in range(instances_per_bag)]))
        labels.append(3)

    return np.array(data), np.array(labels)

if __name__ == "__main__":
    print(generate_data(bags_per_function=10)[0][0])
