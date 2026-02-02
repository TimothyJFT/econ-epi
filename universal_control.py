import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pickle


# INDIVIDUAL
def Nash_condition(a):

    sol = solve_ivp(RHSode_Nash, t_span, y0, args=(a,), t_eval=t_eval)
    S,I = sol.y

    total_susceptibles = S0*np.exp(np.trapz(-beta * a**2 * I, sol.t))
    total_susceptibles = S[-1]

    new_a = (2*control_cost)  / (2*control_cost + infection_cost * beta * I * total_susceptibles)

    return (1-omega)*a + omega*new_a


def converge(initial_guess):

    prev_a = initial_guess
    new_a = Nash_condition(prev_a)

    max_iterations = 1000
    for i in range(max_iterations):

        diff = np.max(np.abs(new_a - prev_a))
        if diff < epsilon:
            #print(i, "iterations to compute Nash equilibrium.")
            return new_a

        prev_a = new_a
        new_a = Nash_condition(prev_a)

    print("Error: Nash solution did not converge!")
    return new_a


def RHSode_Nash(t, y, a): 
    S, I = y

    S = max(0, min(1, S))
    I = max(0, min(1, I))

    a_t = float(np.interp(t, t_eval, a))
    
    dS = -beta * a_t**2 * S * I
    dI = beta * a_t**2 * S * I - gamma * I

    return [dS, dI]


# CENTRAL PLANNER FUNCTIONS
def get_a(S, I, choice):

    a,b = choice

    a = ( 1/(1 + a*S*I) )**b

    a = np.clip(a, 0, 1)

    return a


def RHSode(t, y, choice):
    S, I = y

    a = get_a(S,I, choice)

    dS = -beta * a**2 * S * I
    dI = beta * a**2 * S * I - gamma * I

    return [dS, dI]


# Baseline result with a_S(t) = 1 = a_I(t)
def RHSode_baseline(t, y):
    S, I = y

    S = max(0, min(1, S))
    I = max(0, min(1, I))

    dS = -beta * S * I
    dI = beta * S * I - gamma * I

    return [dS, dI]


def cost_function(choice):

    # Get solution given choice variables
    sol = solve_ivp(RHSode, t_span, y0, t_eval=t_eval, method='RK45',
                    args=(choice, ),
                    rtol=1e-7, atol=1e-9)

    S,I = sol.y

    # Calculate cost
    ## Controls costs
    a = get_a(S, I, choice)

    cost_rate = control_cost* (1-a)**2 
    net_control_cost = np.trapz(cost_rate, sol.t)

    ## Cost of infections (k per)
    net_infection_cost = infection_cost * ( 1 - S0*np.exp(np.trapz(-beta * a**2 * I, sol.t)))
    #net_infection_cost = infection_cost * (1-S[-1])
    
    return net_control_cost + net_infection_cost


def dothings(doprint=True, dowrite=True):
    global y0, t_span, t_eval
        
    with open("parameters.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Execute the assignment in the global namespace
            exec(line, globals())

    y0 = (S0, I0)
    t_span = (0, duration) # Start time, end time
    t_eval = np.linspace(t_span[0], t_span[1], 1+evals)


    baseline_sol = solve_ivp(RHSode_baseline, t_span, y0, t_eval=t_eval, method='RK45')

    initial_guess = np.full_like(t_eval, 0.5)
    Nash_a = converge(initial_guess)

    Nash_sol = solve_ivp(RHSode_Nash, t_span, y0, t_eval=t_eval, args=(Nash_a,), method='RK45')
    Nash_obj = np.trapz(control_cost*(1-Nash_a)**2, Nash_sol.t) + infection_cost * (1-Nash_sol.y[0][-1])

    # Get optimal solution
    alpha = infection_cost*beta / (2*control_cost)
    choice0 = (2*infection_cost*beta/control_cost,1,)

    bounds = ((None,None), (None, None))

    result = minimize(cost_function, choice0, bounds=bounds, method='L-BFGS-B')

    optimal_sol = solve_ivp(RHSode, t_span, y0, t_eval=t_eval, args=(result.x,), method='RK45')
    a = get_a(optimal_sol.y[0], optimal_sol.y[1], result.x) # a(t) trajectory

    # Printing
    if doprint:
        print("BASELINE")
        print("Baseline objective value:",infection_cost * (1-baseline_sol.y[0][-1]))

        print("\nNASH")
        print("Nash objective value:", Nash_obj)

        print("\nCENTRAL PLANNER")
        print("Parameters:")
        print(" coefficient:", result.x[0])
        print(" power:", result.x[1])

        print("\nObjective value:", result.fun)

    if dowrite:
        out = {
            "baseline":baseline_sol,
            "Nash":Nash_sol,
            "centralised":optimal_sol,
            "central_a":a,
            "Nash_a":Nash_a
            }
        with open("univ_out.pkl", "wb") as f:
            pickle.dump(out, f)

    return result.fun / Nash_obj,  np.max(optimal_sol.y[1]) <= I0+epsilon, np.max(Nash_sol.y[1]) <= I0+epsilon


if __name__ == '__main__':
    dothings()
