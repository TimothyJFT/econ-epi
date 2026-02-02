SPECIFIC CASES

AGGREGATE RESULTS

universal_control.py and compcontrol_4params run a single universal/compartmental epidemic simulation based on the parameters.txt file. 

masterfile.py runs many epidemic simulations based on the desired parameters. It rewrites the parameters.txt file many times as it runs, for reasons of historical contingency. It saves its output to recent_results.csv.

heat_mapper.R reads recent_results.csv and (for the desired value of \chi ("x"), with \chi = 1 for the universal case) and produces a heat map.
