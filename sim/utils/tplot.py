import plotext as plt
import pandas as pd
import argparse

"""
terminalplot: plots the time series data from a CSV file using plotext.
Usage: 
python terminalplot.py <filepath> <column1> <column2> ...
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Process a CSV file and extract specified columns.")
    parser.add_argument("sim_id", type=str, help="Id of the simulation to load")
    parser.add_argument("columns", nargs='+', help="Two or more column names")    
    parser.add_argument("--copy_to_clipboard", "-ctc", action='store_true', help="Copy the DataFrame to clipboard")

    args = parser.parse_args()

    # Ensure at least two columns are given
    if len(args.columns) < 2:
        parser.error("At least two column names must be provided.")

    return args

def make_plot(df, x_col, y_cols, width, height):
    plt.clf()
    plt.plotsize(width, height)
    plt.xaxes(1, 0)
    plt.yaxes(1, 0)
    plt.title("Time Series Plot")
    plt.xlabel(x_col)

    
    if len(y_cols) > 1:
        plt.ylabel("Values")
        for col in y_cols:
            x = df[x_col].tolist()
            y = df[col].tolist()
            plt.plot(x, y, label=col)
    else:
        plt.ylabel(y_cols[0])        
        x = df[x_col].tolist()
        y = df[y_cols[0]].tolist()
        plt.plot(x,y)

    return plt.build()

if __name__ == "__main__":
    args = parse_args()

    print(f"Selected Simulation: {args.sim_id}")
    print(f"Column for x-axis: {args.columns[0]}")
    print(f"Columns for line plots: {args.columns[1:]}")

    filepath = "output/" + args.sim_id + "/time-log.csv"

    df = pd.read_csv(filepath, usecols=args.columns)

    if args.copy_to_clipboard:
        df.to_clipboard(index=False)
        print("DataFrame copied to clipboard.")

    make_plot(df, args.columns[0], args.columns[1:], width=100, height=30)
    plt.show()