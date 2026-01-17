import pandas as pd
import numpy as np
import argparse
import json
import csv
from sim.utils.table import print_dataframe


"""
Creates a Cross Table with input params on x and y-axis and values from the time table as values.
Usage example: 
python -m sim.utils.crosstab -sid 4mpi -min 1 -max 11 -time 1.0
python -m sim.utils.crosstab -sid 10conv -min 1 -max 4 -time 1.0
python -m sim.utils.crosstab -sid unit -max 11 -time 0.0

"""

def parse_args():
    parser = argparse.ArgumentParser(description="Reads values of a collection of simulations and returns a cross table.")
    parser.add_argument("-sid", type=str, help="Id of the simulations to load")
    parser.add_argument("-min", type=int, help="Loads all simulations from sim_id+min to sim_id+max.", default = 1)
    parser.add_argument("-max", type=int, help="Loads all simulations from sim_id+min to sim_id+max.")
    parser.add_argument("-key", type=str, help="Key for evaluation of timetable.", default="errorL2")
    parser.add_argument("-time", type=float, help="Time point for evaluation.", default=0.0)
    parser.add_argument("-keyx", type=str, help="Key for x-axis of cross table.", default="dh")
    parser.add_argument("-keyy", type=str, help="Key for y-axis of cross table.", default="dt")
    parser.add_argument("--copy_to_clipboard", "-ctc", action='store_true', help="Copy the DataFrame to clipboard")
    parser.add_argument("-latex", type=int, help="Print Latex Version of table or not.", default = 1)
    parser.add_argument("-csv", type=int, help="Print Latex Version of table or not.", default = 1)

    args = parser.parse_args()
    print(args)

    return args

def get_ranks_from_csv(filename):
    with open(filename) as fd:
        reader=csv.reader(fd)
        interestingrows=[row for idx, row in enumerate(reader) if idx in (0,1)]
    if interestingrows[0][0] == "# MPI Ranks":
        return int(interestingrows[1][0])
    else:
        return "unknown"

if __name__ == "__main__":
    args = parse_args()

    sims = [args.sid+str(i) for i in range(args.min, args.max+1,1)]
    print(f"Selected Simulations: {sims}")
    
    filepaths_log = ["output/" + sim + "/time-log.csv" for sim in sims]
    filepaths_config = ["output/" + sim + "/config.json" for sim in sims]

    
    rows = []
    cols = []
    vals = []
    for i in range(len(sims)):
        sim = sims[i]
        try:
            df = pd.read_csv("output/" + sim + "/time-log.csv", usecols=["time", args.key])
            val = df.loc[np.isclose(df["time"],args.time), args.key].item()
            vals.append(val)

            with open("output/" + sim + "/config.json", "r") as f:
                config = json.load(f)
            if args.keyy != "n":
                rows.append(config[args.keyy])
            else:
                rows.append(get_ranks_from_csv("output/" + sim + "/statics.csv"))

            if args.keyx != "n":
                cols.append(config[args.keyx])
            else:
                cols.append(get_ranks_from_csv("output/" + sim + "/statics.csv"))
        
        except Exception as e:
            print("Warning: not able to process simulation ", sims[i], " due to Exception: ", e)
    
    
    df_out = pd.crosstab(rows, cols, values=vals, aggfunc="sum", rownames = [args.keyy], colnames = [args.keyx])
    
    if args.copy_to_clipboard:
        df_out.to_clipboard(index=False)
        print("DataFrame copied to clipboard.")

    if args.latex == 1:
        print(df_out.to_latex())
    
    if args.csv == 1:
        print(df_out.to_csv())
        
    print_dataframe(df_out.reset_index())