import pandas as pd
import numpy as np
import argparse
import json
from rich.console import Console
from rich.table import Table
from sim.utils.table import print_dataframe

"""
Merges a column of the time log over several simulations to one table.

Usage example: 
python -m sim.utils.merge -sid spiralcomp -max 4 -time 1.0
"""



def parse_args():
    parser = argparse.ArgumentParser(description="Reads values of a collection of simulations and returns a cross table.")
    parser.add_argument("-sid", type=str, help="Id of the simulations to load")
    parser.add_argument("-min", type=int, help="Loads all simulations from sim_id+min to sim_id+max.", default = 1)
    parser.add_argument("-max", type=int, help="Loads all simulations from sim_id+min to sim_id+max.")
    parser.add_argument("-col", type=str, help="Column key for evaluation of timetable.", default="errorL2")
    parser.add_argument("--copy_to_clipboard", "-ctc", action='store_true', help="Copy the DataFrame to clipboard")
    parser.add_argument("-latex", type=int, help="Print Latex Version of table or not.", default = 1)
    parser.add_argument("-csv", type=int, help="Print Latex Version of table or not.", default = 1)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    sims = [args.sid+str(i) for i in range(args.min, args.max+1,1)]
    print(f"Selected Simulations: {sims}")
    
    filepaths_log = ["output/" + sim + "/time-log.csv" for sim in sims]



    df = pd.DataFrame()

    for i in range(len(sims)):
        filepath_log = filepaths_log[i]
        try:
            new_df  = pd.read_csv(filepath_log, usecols=["time", args.col], index_col="time").rename(columns={args.col: sims[i]+"."+args.col})
            df      = pd.concat([df, new_df], axis = 1)

        except Exception as e:
            print("Warning: not able to process simulation ", sims[i], " due to Exception: ", e)
            
    if args.copy_to_clipboard:
        df.to_clipboard(index=False)
        print("DataFrame copied to clipboard.")

    if args.latex == 1:
        print(df.to_latex())

    if args.csv == 1:
        print(df.to_csv())

    print_dataframe(df.reset_index(), title=args.sid)

