import pandas as pd
import argparse
from rich.console import Console
from rich.table import Table

"""
table: plots the data from a CSV file using rich terminal.
Usage for all columns: 
python -m sim.utils.table ldpg_smooth -ctc
Usage for specific columns:
python -m sim.utils.table ldpg_smooth "Energies (elastic)" -ctc
"""

def print_dataframe(df: pd.DataFrame, title: str = "DataFrame") -> None:
    table = Table(title=title)

    # Add column headers
    for col in df.columns:
        table.add_column(str(col))

    # Add rows
    for _, row in df.iterrows():
        table.add_row(*[str(cell) for cell in row])

    console = Console()
    console.print(table)



def parse_args():
    parser = argparse.ArgumentParser(description="Process the time-log of a specific simulation and extract specified columns.")
    parser.add_argument("sim_id", type=str, help="Id of the simulation to load")
    parser.add_argument("columns", nargs='*', help="Two or more column names")
    parser.add_argument("--copy_to_clipboard", "-ctc", action='store_true', help="Copy the DataFrame to clipboard")
    parser.add_argument("-latex", type=int, help="Print Latex Version of table or not.", default = 0)
    parser.add_argument("-csv", type=int, help="Print Latex Version of table or not.", default = 1)

    args = parser.parse_args()
    print(args)

    return args


if __name__ == "__main__":
    args = parse_args()
    print(f"Selected Simulation: {args.sim_id}")
    
    filepath = "output/" + args.sim_id + "/time-log.csv"

    if len(args.columns) > 0:
        print(f"Columns to display: {args.columns}")
        df = pd.read_csv(filepath, usecols=args.columns)
    else:
        df = pd.read_csv(filepath)
    
    if args.copy_to_clipboard:
        df.to_clipboard(index=False)
        print("DataFrame copied to clipboard.")

    if args.latex == 1:
        print(df.to_latex())
    
    if args.csv == 1:
        print(df.to_csv())

    print_dataframe(df, title=args.sim_id)