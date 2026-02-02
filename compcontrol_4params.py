import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pickle


# INDIVIDUAL
def Nash_condition(a):

    sol = solve_ivp(RHSode_Nash, t_span, y0, args=(a,), t_eval=t_eval,)
    S,I = sol.y

    total_susceptibles = S0*np.exp(np.trapz(-beta * a**2 * I, sol.t))
    total_susceptibles = S[-1]

    # Simplified
    simple_a = 1 - infection_cost * beta * I * total_susceptibles / (2*control_cost*S)

    # Correction
    complex_a = np.full_like(simple_a, 0)
    complex_a[-1] = simple_a[-1] # 1 - infection_cost * beta * I[-1] * total_susceptibles / (2*control_cost*S[-1]) # 

    ## Trapezium
    #for i in range(evals-1,-1,-1): # Starting at one tick before the final eval, iterate backwards.
    #    integral = np.trapz(S[i+1:] * (1-complex_a[i+1:])**2, t_eval[i+1:])
    #    complex_a[i] = simple_a[i] +  (beta*I[i]/(2*S[i])) * integral
    ## Right-hand Riemann
    dt = duration/evals
    for i in range(evals-1,-1,-1): # Starting at one tick before the final eval, iterate backwards.
        integral = np.sum(S[i+1:evals+1] * (1 - complex_a[i+1:evals+1])**2) * dt
        complex_a[i] = simple_a[i] +  (beta*I[i]/(2*S[i])) * integral

    #print(np.max(np.abs(simple_a - complex_a)))
    return (1-omega)*a + omega * complex_a


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
    print(diff)
    return new_a


def RHSode_Nash(t, y, a): # Note a_I(t) = 1
    S, I = y

    S = max(0, min(1, S))
    I = max(0, min(1, I))

    a_t = float(np.interp(t, t_eval, a))
    
    dS = -beta * a_t * S * I
    dI = beta * a_t * S * I - gamma * I

    return [dS, dI]


# CENTRAL PLANNER FUNCTIONS
def get_aSI(S, I, choice):
    a, b, d, e = choice

    # Make array-like; idea is that the function works if passed floats or an array of floats
    S = np.asarray(S)
    I = np.asarray(I)

    aS = ((1 - a*I) / (1 - a*b*S*I))
    dxS = np.minimum(d*S, 1)
    aI = ((1 - dxS) / (1 - dxS*e*I))

    # Clip aS between 0 and 1
    aS = np.clip(aS, 0, 1)

    aI = np.clip(aI, aS*x_min, 1)

    # If inputs were scalars, return scalars
    if np.isscalar(S) and np.isscalar(I):
        return float(aS), float(aI)
    return aS, aI


def RHSode_central(t, y, choice):
    S, I = y

    aS,aI = get_aSI(S,I, choice)

    dS = -beta * aS*aI * S * I
    dI = beta * aS*aI * S * I - gamma * I

    return [dS, dI]


# Baseline result with a_S(t) = 1 = a_I(t)
def RHSode_baseline(t, y):
    S, I= y

    S = max(0, min(1, S))
    I = max(0, min(1, I))

    dS = -beta * S * I
    dI = beta * S * I - gamma * I

    return [dS, dI]


def cost_function(choice):

    # Get solution given choice variables
    sol = solve_ivp(RHSode_central, t_span, y0, t_eval=t_eval, method='RK45',
                    args=(choice, ),
                    rtol=1e-7, atol=1e-9)

    S,I = sol.y

    # Calculate cost
    ## Control costs
    aS,aI = get_aSI(S, I, choice)
    
    cost_rate = control_cost* S*(1-aS)**2 + control_cost* I*(1-aI)**2 
    net_control_cost = np.trapz(cost_rate, sol.t)

    ## Cost of infections (k per)
    net_infection_cost = infection_cost * ( 1 - S0*np.exp(np.trapz(-beta * aS*aI * I, sol.t)))
    #net_infection_cost = infection_cost * (1-S[-1])# Alterative specificatino. Shouldn't matter much.
    
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
    Nash_obj = control_cost*np.trapz(Nash_sol.y[0]*(1-Nash_a)**2, Nash_sol.t) + infection_cost * (1-Nash_sol.y[0][-1])

    # Get optimal solution
    alpha = infection_cost*beta / (2*control_cost)
    choice0 = (alpha, alpha, alpha, alpha)

    bounds = ((None,None), (0,1), (None,None), (0,1))

    result = minimize(cost_function, choice0, bounds=bounds, method='L-BFGS-B')

    optimal_sol = solve_ivp(RHSode_central, t_span, y0, t_eval=t_eval, args=(result.x,), method='RK45')
    aS,aI = get_aSI(optimal_sol.y[0], optimal_sol.y[1], result.x) # Trajectories of a_j(t)

    # Printing
    if doprint:
        print("BASELINE")
        print("Baseline objective value:",infection_cost * (1-baseline_sol.y[0][-1]))

        print("\nNASH")
        print("Nash objective value:", Nash_obj)

        print("\nCENTRAL PLANNER")
        print("S Parameters:")
        print(" numerator coefficient:", result.x[0])
        print(" denominator multiplier:", result.x[1])

        print("I parameters:")
        print(" numerator coefficient:", result.x[2])
        print(" denominator multiplier:", result.x[3])

        print("\nObjective value:", result.fun)

    if dowrite:
        out = {
            "baseline":baseline_sol,
            "Nash":Nash_sol,
            "centralised":optimal_sol,
            "central_aS":aS,
            "central_aI":aI,
            "Nash_aS":Nash_a
            }
        with open("comp_out.pkl", "wb") as f:
            pickle.dump(out, f)

    return result.fun / Nash_obj, np.max(optimal_sol.y[1]) <= I0+epsilon, np.max(Nash_sol.y[1]) <= I0+epsilon

if __name__ == '__main__':
    dothings()
