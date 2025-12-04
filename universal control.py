import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Parameters
beta = 0.5
gamma = 0.2
control_cost = 200
infection_cost = 10000

epsilon = 1e-12 # For rounding purposes

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

    new_a = (2*control_cost)  / (2*control_cost + infection_cost * beta * I * total_susceptibles)

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


def RHSode_Nash(t, y, a): 
    S, I = y

    #S = max(0, min(1, S))
    #I = max(0, min(1, I))

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

    #S = max(0, min(1, S))
    #I = max(0, min(1, I))

    a = get_a(S,I, choice)

    dS = -beta * a**2 * S * I
    dI = beta * a**2 * S * I - gamma * I

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


# PLOTTING

print("BASELINE")
baseline_sol = solve_ivp(RHSode_baseline, t_span, y0, t_eval=t_eval, method='RK45')
print("Baseline objective value:",infection_cost * (1-baseline_sol.y[0][-1]))

print("\nNASH")
initial_guess = np.full_like(t_eval, 0.3)
Nash_a = converge(initial_guess)

Nash_sol = solve_ivp(RHSode_Nash, t_span, y0, t_eval=t_eval, args=(Nash_a,), method='RK45')
print("Nash objective value:",infection_cost * (1-Nash_sol.y[0][-1]))

# Get optimal solution
alpha = infection_cost*beta / (2*control_cost)
#choice0 = (alpha, alpha**2, 1)
choice0 = (2*infection_cost*beta/control_cost,1,) # Or something?

bounds = ((None,None), (None, None))

result = minimize(cost_function, choice0, bounds=bounds, method='L-BFGS-B')

print("\nCENTRAL PLANNER")
print("Parameters:")
print(" coefficient:", result.x[0])
print(" power:", result.x[1])

print("\nObjective value:", result.fun)

optimal_sol = solve_ivp(RHSode, t_span, y0, t_eval=t_eval, args=(result.x,), method='RK45')

# Plotting
plt.plot(optimal_sol.t, optimal_sol.y[0], label='Central Planner Susceptible (S)', color='blue')
plt.plot(optimal_sol.t, optimal_sol.y[1], label='Central Planner Infected (I)', color='red')

plt.plot(Nash_sol.t, Nash_sol.y[0], label='Nash Susceptible (S)', color='cyan')
plt.plot(Nash_sol.t, Nash_sol.y[1], label='Nash Infected (I)', color='orange')

plt.plot(baseline_sol.t, baseline_sol.y[0], label='Baseline Susceptible (S)', linestyle='--', color='blue')
plt.plot(baseline_sol.t, baseline_sol.y[1], label='Baseline Infected (I)', linestyle='--', color='red')

# Compute a_j(t) trajectories
a = get_a(optimal_sol.y[0], optimal_sol.y[1], result.x)
plt.plot(optimal_sol.t, a, label='Central Planner a(t)', color='purple')

plt.plot(Nash_sol.t, Nash_a, label='Nash a(t)', color='pink')

plt.xlabel('Time')
plt.ylabel('Proportion / Control')
plt.title('Epidemic with central planner and individual solutions')
plt.legend()
plt.grid(True)

plt.ylim(-0.1,1.1)

plt.show()
