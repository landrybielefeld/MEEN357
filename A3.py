import numpy as np
import math
import matplotlib.pyplot as plt

# ================= METHODS ==================

def bisection(func, lb, ub, err_max='null', iter_max=1000):

    if (type(lb) != int and type(lb) != float) or (type(ub) != int and type(ub) != float):
        raise Exception('Upper and Lower Bound Variables must be integer or float') 
    if (type(iter_max) != int or iter_max <= 0):
        raise Exception('iter max must be a positive integer')
    if (type(err_max) != int and type(err_max) != float and err_max != 'null'):
        raise Exception('err_max must be a integer or float')
    #! This does not work when it comes to some funcitons. 
    if (not func(lb) * func(ub) < 0):
        raise Exception('No root found between the given bounds')

    xr = 0
    err_rel = -1         # store the relative approximate error to the last estimation
    iter = 0            # start at iteration 0
    exit_flag = 1       # Assume true flag at the beginning

    # Set guesses to upper and lower bounds
    xl = lb
    xu = ub

    func_tol = 0.0001    # This is the tolerance of the function around 0

    # In the case the chosen upper or lower bounds happen to be the root
    # err_rel will return -1 since this was never calculated
    if abs(func(xl)) < func_tol:
        return xl, err_rel, iter, 1
    if abs(func(xu)) < func_tol:
        return xu, err_rel, iter, 1


    done = False
    while not done:
        xr = (xu + xl) / 2.0    # solve for the middle value

        # check to see if max iteration is hit
        if iter == iter_max:
            return xr, err_rel, iter, 0
        # return and check if no root is found in the current bracket
        if (func(xl) * func(xu) > 0):
            return xr, err_rel, iter, -1
        # check to see if invalid return value from function
        if np.isnan(func(xl)) or np.isnan(func(xr)) or np.isinf(func(xl)) or np.isinf(func(xr)):
            return xr, err_rel, iter, -2

        # Set new upper or lower bound
        if func(xl) * func(xr) < 0:
            xu = xr
        else:
            xl = xr # simplified if statement    
            
        # Check to see if root is found or caps are hit
        if err_max != 'null':
            err_rel = 100 * abs((xl - xu)/(xu + xl))    # Solve for relative error (%)
            if err_rel < err_max:
                done = True

        # If iterations max is hit or the root is found...
        if iter >= iter_max or abs(func(xr)) < func_tol:
            done = True

        iter += 1

    # print('Done computing. Found solution in {:d} iterations.\n'.format(iter))
    # print('Estimated root location {:.4f}\n'.format(xr))
    # if err_max != 'null':
    #     print('Estimated relative error = {:.4f}\n'.format(err_rel))

    return xr, err_rel, iter, 1

def falsepos(func, lb, ub, err_max='null', iter_max=1000):

    if (type(lb) != int and type(lb) != float) or (type(ub) != int and type(ub) != float):
        raise Exception('Upper and Lower Bound Variables must be integer or float') 
    if (type(iter_max) != int or iter_max <= 0):
        raise Exception('iter max must be a positive integer')
    if (type(err_max) != int and type(err_max) != float and err_max != 'null'):
        raise Exception('err_max must be a integer or float')
    #! This does not work when it comes to some funcitons. 
    if (not func(lb) * func(ub) < 0):
        raise Exception('No root found between the given bounds')

    xr = 0
    err_rel = -1         # store the relative approximate error to the last estimation
    iter = 0            # start at iteration 0
    exit_flag = 1       # Assume true flag at the beginning

    # Set guesses to upper and lower bounds
    xl = lb
    xu = ub

    func_tol = 0.0001    # This is the tolerance of the function around 0

    # In the case the chosen upper or lower bounds happen to be the root
    # err_rel will return -1 since this was never calculated
    if abs(func(xl)) < func_tol:
        return xl, err_rel, iter, 1
    if abs(func(xu)) < func_tol:
        return xu, err_rel, iter, 1


    done = False
    while not done:
        xr = xu - (func(xu) * (xl - xu)) / (func(xl) - func(xu))

        # check to see if max iteration is hit
        if iter == iter_max:
            return xr, err_rel, iter, 0
        # return and check if no root is found in the current bracket
        if (func(xl) * func(xu) > 0):
            return xr, err_rel, iter, -1
        # check to see if invalid return value from function
        if np.isnan(func(xl)) or np.isnan(func(xr)) or np.isinf(func(xl)) or np.isinf(func(xr)) or np.isnan(func(xu)) or np.isnan(func(xu)):
            return xr, err_rel, iter, -2

        # Set new upper or lower bound
        if func(xl) * func(xr) < 0:
            xu = xr
        else:
            xl = xr # simplified if statement    
            
        # Check to see if root is found or caps are hit
        if err_max != 'null':
            err_rel = 100 * abs((xl - xu)/(xu + xl))    # Solve for relative error (%)
            if err_rel < err_max:
                done = True

        # If iterations max is hit or the root is found...
        if iter >= iter_max or abs(func(xr)) < func_tol:
            done = True

        iter += 1

    return xr, err_rel, iter, 1

def secant(func, x1, err_max='null', iter_max=1000):

    if (type(x1) != int and type(x1) != float):
        raise Exception('Guess Variable must be integer or float') 
    if (type(iter_max) != int or iter_max <= 0):
        raise Exception('iter max must be a positive integer')
    if (type(err_max) != int and type(err_max) != float and err_max != 'null'):
        raise Exception('err_max must be a integer or float')

    err_rel = -1         # store the relative approximate error to the last estimation
    iter = 0             # start at iteration 0
    x0 = x1 - .2          # Creates second guess a little off
    x2 = 0               # This wll store the stored value if x2 xi+1
    x2_prev = 0          # This stores the last value of x2 for the calculation of the approximate rel error

    func_tol = 0.0001    # This is the tolerance of the function around 
    # In the case the guess happens to be the root
    # err_rel will return -1 since this was never calculated
    if abs(func(x1)) < func_tol:
        return x1, err_rel, iter, 1

    done = False
    while not done:
        # check to see if invalid return value from function
        if np.isnan(func(x1)) or np.isnan(func(x0)) or np.isinf(func(x1)) or np.isinf(func(x0)):
            return x2, err_rel, iter, -2    # x2 won't mean much in this case

        x2 = x1 - (func(x1)*(x0 - x1))/(func(x0) - func(x1))    # set x2 to the possible root

        # check to see if max iteration is hit
        if iter == iter_max:
            return x2, err_rel, iter, 0 
            
        # Check to see if root is found or caps are hit
        if err_max != 'null' and iter > 0:
            err_rel = 100 * abs((x2 - x2_prev)/(x2))    # Solve for relative error (%)
            if err_rel < err_max:
                done = True

        # If iterations max is hit or the root is found...
        if iter >= iter_max or abs(func(x2)) < func_tol:
            done = True

        x0 = x1
        x1 = x2
        x2_prev = x2
        iter += 1

    return x2, err_rel, iter, 1


#================ FUNCTIONS =================

def func1(x):
    return x * math.sin(x) + 3 * math.cos(x) - x

def func2(x):
    return x * (math.sin(x) - x * math.cos(x))

def func3(x):
    return (math.pow(x, 3) - 2 * math.pow(x, 2) + 5 * x - 25)/40
    
def findRoots(lr, ur, funcs):
    # Loop through the functions
    # Function 1: Using False position method
    # Funciton 2 & 3: Using secant method
    for i in range(len(funcs)):
        if i == 0:
            step = .2           # The search size to find all the roots in steps
            xl = lr             # set lower bound to -6 in this case
            xu = lr + step      # set upper bound
            func = funcs[i]
            roots = []          # will store all the roots

            safe = 0            # variable to keep the while loop from being infinite
            while xu < ur or safe == 50:
                if func(xu) * func(xl) < 0:
                    root, err_rel, iter, flag = falsepos(func, xl, xu)
                    xl = xu # shift the search a bit
                    if flag == 1:
                        roots.append(root)
                else:
                    xu = xu + step      # set upper bound
                safe += 1
            print("The roots found for the 1st function using the false position method was:")
            print(roots)
            print()
        if i == 1:
            step = .2           # The search size to find all the roots in steps
            xl = lr             # set lower bound to -6 in this case
            xu = lr + step      # set upper bound
            func = funcs[i]     # grab the funciton needed
            roots = []          # will store all the roots

            safe = 0            # variable to keep the while loop from being infinite
            while xu < ur or safe == 1000:
                xl = xl + step  # shift the secant line to search by a step size
                xu = xu + step
                root, err_rel, iter, flag = secant(func, xl)
                if flag == 1:
                    # If root is found, need to make sure it hasn't already been added
                    hasRootAlreadyBeenFound = False
                    rootTol = .5
                    for r in roots:
                        if abs(root - r) < rootTol or root < lr or root > ur:
                            hasRootAlreadyBeenFound = True  # if root is out of bounds or has already been found, don't add it
                    if not hasRootAlreadyBeenFound:
                        roots.append(root)  # add root
                safe += 1   # increment the safety index
            print("The roots found for the 2nd function using the secant method was:")
            print(roots)
            print()
        if i == 2:
            xl = lr         # This function is simply third order so the root can be found with one secant search
            roots = []      # clear the roots to add new ones
            func = funcs[i]
            root, err_rel, iter, flag = secant(func, xl)
            if flag == 1:
                roots.append(root)  # add root
            safe += 1   # increment the safety index
            print("The roots found for the 1st function using the secant method was:")
            print(roots)
                    
findRoots(-6, 6, [func1, func2, func3])

#================ MAIN =====================
# print('Bisection Method Results:')
# print(bisection(func1, -5, 5, .001))
# #print(bisection(func2, -10, 10, .01))
# print(bisection(func3, -5, 5, .001))


# print('\nFalse Position Method Results:')
# print(falsepos(func1, -5, 5, .001))
# #print(falsepos(func2, -10, 10, .01))
# print(falsepos(func3, -5, 5, .001))


# print('\nSecant Method Results:')
# print(secant(func1, 5, .001))
# print(secant(func2, 5, .001))
# print(secant(func3, 5, .001))


