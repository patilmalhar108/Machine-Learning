from numpy import arange
from numpy import asarray
from numpy.random import rand
from matplotlib import pyplot

def obj(x):
    return x**2.0

def derv(x):
    return x*2.0

def grad(obj, derv, bounds, n_iteration, step_size):
    solution, score = list(), list()
    solution = bounds[:,0] + rand(len(bounds)) * (bounds[:,1] - bounds[:,0])
    for i in range(n_iteration):
        grad = derv(solution)
        solution = solution - step_size * grad
        solution_ev = obj(solution)
        score.append(solution_ev)
        print(">%d f(%s) = %.5f" %(i, solution, solution_ev))
    return [solution, score]

bounds = asarray([[-1.0,1.0]])
n_iteration = 30
step_size = 0.5

solution, score = grad(obj, derv, bounds, n_iteration, step_size)
inputs = arange(bounds[0,0], bounds[0,1] + 0.1,0.1)
result = obj(inputs)
pyplot.plot(inputs, result)
pyplot.plot(solution, score, '.-', color = "red")
pyplot.show()