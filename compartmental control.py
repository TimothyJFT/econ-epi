import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Parameters
beta = 0.5
gamma = 0.2
control_cost = 200
infection_cost = 1000

epsilon = 1e-12 # For rounding purposes

x_min = 0.8

## Nash parameters
omega = 0.5

# ICs
S0 = 0.99
I0 = 1-S0
y0 = (S0,I0)

# Time
duration = 100
evals = 1000
t_span = (0, duration) # Start time, end time
t_eval = np.linspace(t_span[0], t_span[1], 1+evals)


# INDIVIDUAL
def Nash_condition(a):

    sol = solve_ivp(RHSode_Nash, t_span, y0, args=(a,), t_eval=t_eval,)
    S,I = sol.y

    total_susceptibles = S0*np.exp(np.trapz(-beta * a**2 * I, sol.t))
    total_susceptibles = S[-1]

    new_a = 1 - infection_cost * beta * I * total_susceptibles / (2*control_cost*S)

    return (1-omega)*a + omega*new_a


def converge(initial_guess):

    prev_a = initial_guess
    new_a = Nash_condition(prev_a)

    max_iterations = 1000
    for i in range(max_iterations):

        diff = np.max(np.abs(new_a - prev_a))
        if diff < epsilon:
            print(i, "iterations to compute Nash equilibrium.")
            return new_a

        prev_a = new_a
        new_a = Nash_condition(prev_a)

    print("Error: Nash solution did not converge!")
    return new_a


def RHSode_Nash(t, y, a): # Note a_I(t) = 1
    S, I = y

    #S = max(0, min(1, S))
    #I = max(0, min(1, I))

    a_t = float(np.interp(t, t_eval, a))
    
    dS = -beta * a_t * S * I
    dI = beta * a_t * S * I - gamma * I

    return [dS, dI]


# CENTRAL PLANNER FUNCTIONS
def get_aSI(S, I, choice): # Rewritten by Copilot
    a, b, c, d, e, f = choice

    # Ensure array-like for consistent broadcasting
    S = np.asarray(S)
    I = np.asarray(I)

    aS = ((1 - a*I) / (1 - a*b*S*I))**c
    dxS = np.minimum(d*S, 1)
    aI = ((1 - dxS) / (1 - dxS*e*I))**f

    # Clip aS between 0 and 1
    aS = np.clip(aS, 0, 1)

    # Elementwise mask: if aS < x_min, set aI = aS
    #mask = aS < x_min
    #aI = np.where(mask, aS, aI)

    # Clip aI between x_min and 1
    aI = np.clip(aI, aS*x_min, 1)

    # If inputs were scalars, return scalars
    if np.isscalar(S) and np.isscalar(I):
        return float(aS), float(aI)
    return aS, aI


def RHSode_central(t, y, choice):
    S, I = y

    #S = max(0, min(1, S))
    #I = max(0, min(1, I))

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
    ## Controls costs
    aS,aI = get_aSI(S, I, choice)
    
    cost_rate = control_cost* S*(1-aS)**2 + control_cost* I*(1-aI)**2 
    net_control_cost = np.trapz(cost_rate, sol.t)

    ## Cost of infections (k per)
    net_infection_cost = infection_cost * ( 1 - S0*np.exp(np.trapz(-beta * aS*aI * I, sol.t)))
    #net_infection_cost = infection_cost * (1-S[-1])
    
    return net_control_cost + net_infection_cost


# PLOTTING

print("BASELINE")
baseline_sol = solve_ivp(RHSode_baseline, t_span, y0, t_eval=t_eval, method='RK45')
print("Baseline objective value:",infection_cost * (1-baseline_sol.y[0][-1]))

print("\nNASH")
initial_guess = np.full_like(t_eval, 0.5)
Nash_a = converge(initial_guess)

Nash_sol = solve_ivp(RHSode_Nash, t_span, y0, t_eval=t_eval, args=(Nash_a,), method='RK45')
print("Nash objective value:",infection_cost * (1-Nash_sol.y[0][-1]))

# Get optimal solution
alpha = infection_cost*beta / (2*control_cost)
choice0 = (alpha, alpha, 1, alpha, alpha, 1)

bounds = ((None,None), (0,1), (None, None), (None,None), (0,1), (None, None))

result = minimize(cost_function, choice0, bounds=bounds, method='L-BFGS-B')

print("\nCENTRAL PLANNER")
print("S Parameters:")
print(" numerator coefficient:", result.x[0])
print(" denominator multiplier:", result.x[1])
print(" power:", result.x[2])

print("I parameters:")
print(" numerator coefficient:", result.x[3])
print(" denominator multiplier:", result.x[4])
print(" power:", result.x[5])

print("\nObjective value:", result.fun)

optimal_sol = solve_ivp(RHSode_central, t_span, y0, t_eval=t_eval, args=(result.x,), method='RK45')

# Plotting
plt.plot(optimal_sol.t, optimal_sol.y[0], label='Central Planner Susceptible (S)', color='blue')
plt.plot(optimal_sol.t, optimal_sol.y[1], label='Central Planner Infected (I)', color='red')

plt.plot(Nash_sol.t, Nash_sol.y[0], label='Nash Susceptible (S)', color='cyan')
plt.plot(Nash_sol.t, Nash_sol.y[1], label='Nash Infected (I)', color='orange')

plt.plot(baseline_sol.t, baseline_sol.y[0], label='Baseline Susceptible (S)', linestyle='--', color='blue')
plt.plot(baseline_sol.t, baseline_sol.y[1], label='Baseline Infected (I)', linestyle='--', color='red')

# Compute a_j(t) trajectories
aS,aI = get_aSI(optimal_sol.y[0], optimal_sol.y[1], result.x)
plt.plot(optimal_sol.t, aS, label='Central Planner a_S(t)', color='purple')
plt.plot(optimal_sol.t, aI, label='Central Planner a_I(t)', color='magenta')

plt.plot(Nash_sol.t, Nash_a, label='Nash a_S(t)', color='pink')

plt.xlabel('Time')
plt.ylabel('Proportion / Control')
plt.title('Epidemic with central planner and individual solutions')
plt.legend()
plt.grid(True)

plt.ylim(-0.1,1.1)

plt.show()
