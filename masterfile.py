import csv
import compcontrol_4params
import universal_control


# I save the parameters as a txt file for reasons of historical contingency.
def set_params(beta, gamma, cc, ic, x_min):
    text = f"""# Parameters
beta = {beta}
gamma = {gamma}
control_cost = {cc}
infection_cost = {ic}

# For rounding purposes
epsilon = 1e-12

# Only for compartmental control
x_min = {x_min}

# Nash parameter ("to promote convergence")
omega = 0.5

# ICs
S0 = 0.99
I0 = 0.01

# Time
duration = 100
evals = 1000
"""
    file = open("parameters.txt",mode='w')
    file.write(text)
    file.close()


results = []

count = 0

# Set parameters to observe.
kclist = [i for i in range(1,12+1)]
Rlist= [(x/20, 0.2) for x in range(3,15+1)]
xlist = [0.5]

print("Main loop started")
for kc in kclist:
    for R in Rlist:
        R0 = round(R[0] / R[1], 2)

        # Universal case (record as x = 1, which is not quite accurate)
        set_params(*R, 1, kc, 1)
        univ_ratio, sup_c, sup_d = universal_control.dothings(doprint=False, dowrite=False)

        results.append({
            "kc": kc,
            "R0": R0,
            "x": 1,
            "ratio": univ_ratio,
            "suppress_cent": sup_c,
            "suppress_dece": sup_d
        })

        # Compartmental cases
        for x in xlist:
            set_params(*R, 1, kc, x)
            comp_ratio,sup_c,sup_d = compcontrol_4params.dothings(doprint=False, dowrite=False)

            results.append({
                "kc": kc,
                "R0": R0,
                "x": x,
                "ratio": comp_ratio,
                "suppress_cent": sup_c,
                "suppress_dece":sup_d
                })

        count += 1
        if count % 10 == 0:
            print(count, 'lots done')

# Store to CSV
with open("recent_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["kc", "R0", "x", "ratio", "suppress_cent","suppress_dece"])
    writer.writeheader()
    writer.writerows(results)
