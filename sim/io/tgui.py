from datetime import datetime
import os, sys
from mpi4py import MPI
import pandas as pd
import numpy as np

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TimeElapsedColumn, TextColumn

class terminal_screen:
    """
    Simple class for a minimalistic terminal screen using the rich library. 
    A case distinction is made to carry over most properties even when mpirun is used. Dynamically sizing e.g. is however lost.
    width and height are only used if dynamically reading the terminal size is not available.
    """
    def __init__(self, dt,T,  refresh_per_second=4, decimal_places=4, width = 200, height =(3+18+7+15)):          
        """
        height = padding + message log + tables
        """

        # Check if standard terminal is available:
        # if script is called with mpirun --> not the case. Layout needs to be refreshed manually        
        if os.isatty(sys.stdout.fileno()):
            self.refresh_manually = False
        else: 
            self.refresh_manually = True

        # Setup console and layout
        if self.refresh_manually:
            self.console =  Console(width= width, height = height)
        else:
            self.console = Console()

        
        # Standard attributes
        self.decimal_places = decimal_places
        self.dt = dt
        self.T = T
        self.height = height
        self.width = width
        self.log_messages = []
        self.static_dict = {}
        self.df_t = pd.DataFrame(columns=["time"])  # Initialize with a time column
        
        # Create Layout
        self.progress = Progress(
            TextColumn("[bold green]Progress:"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True,
            console=self.console,
        )
        self.task_id = self.progress.add_task("Simulation", total=self.T)
        self.static_table = Table(title="Static Values") # left table
        self.time_table = Table(title="Time Statistics") # right table
        self.layout = Layout()
        # define layout
        self.layout.split_column(
            Layout(name="padding",  size = 12) if self.refresh_manually else None,
            Layout(Panel(self.progress, title="Progress", border_style="yellow"), name="progress", size=3),
            Layout(name="log", size=7),
            Layout(name="main", ratio=self.height-7-12)
        )
        self.layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        # start Live view of Layout in the case of only one rank
        if not self.refresh_manually:
            self.live = Live(self.layout, console=self.console, refresh_per_second=refresh_per_second, transient = True)         
            self.live.start()

        self.update_log("Terminal GUI initialized. Manual refreshment due to mpirun necessary: "+str(self.refresh_manually)+ ". # MPI Ranks: "+str(MPI.COMM_WORLD.size)+".") # 

    def manual_refresh(self):
        self.console.print(self.layout)

    def update_log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S") + ", Rank: "+str(MPI.COMM_WORLD.rank)

        # rank wide save
        self.log_messages.append(f"[{timestamp}] {message}")
        
        text = "\n".join(self.log_messages[-5:])
        self.layout["log"].update(Panel(text, title="Log (rank:"+str(MPI.COMM_WORLD.rank)[:]+")", border_style="green"))
        
        if self.refresh_manually: self.manual_refresh()

    def update(self, update_type, data: dict):
        
        current_rank = str(MPI.COMM_WORLD.rank)[:]

        if update_type == "time":
            #update t
            self.progress.update(self.task_id, completed = data["time"])
            # turn nested dictionaries into unnested ones
            normalized_data = pd.json_normalize(data, sep=".")
            # local update
            self.df_t = normalized_data.set_index("time").combine_first(self.df_t.set_index("time") ).reset_index(names=["time"]).sort_values(by="time")

            # self.sync_time_table_over_ranks()

            output_df = self.df_t.round(self.decimal_places)                                # Runden auf 4 nachkommastellen
            output_df = extract_rows_from_df(output_df, every_n= max(int(self.T/self.dt/10),1))    # extract first row, the last five rows and every 10% of the time evolution
            
            self.layout["left"].update(Panel(dataframe_to_rich_table( output_df ),title="Time dependent statistics (rank:"+current_rank+")", border_style="red")) 

        elif update_type == "static":
            self.static_dict.update(data)
            # self.sync_static_table_over_ranks()
            self.layout["right"].update( Panel(dict_to_rich_table(self.static_dict),title="Statics (rank:"+current_rank+")", border_style="blue"))

        if self.refresh_manually: self.manual_refresh()        

def dataframe_to_rich_table(df: pd.DataFrame) -> Table:
    table = Table()

    # Add columns with column headers
    for col in df.columns:
        table.add_column(str(col))

    # Add rows (convert values to str to avoid type issues)
    for _, row in df.iterrows():
        table.add_row(*[str(val) for val in row])

    return table
        
def dict_to_rich_table(data: dict) -> Table:
    table = Table()
    table.add_column("Key", style="bold cyan")
    table.add_column("Value", style="magenta")

    for key, value in data.items():
        table.add_row(str(key), str(value))

    return table

def extract_rows_from_df(df: pd.DataFrame, every_n: int = 100) -> pd.DataFrame:
    # Get index positions
    idx_first = [0]
    idx_nth = list(range(every_n, max(len(df) - 5, 1), every_n))
    idx_last = list(range(max(len(df) - 5, 0), len(df)))

    # Combine and drop duplicates (in case of overlap)
    unique_indices = sorted(set(idx_first + idx_nth + idx_last))

    return df.iloc[unique_indices]


if __name__ == "__main__":
    import time
    """
    Test case
    """
    dt = 0.01
    n_send = 100

    comm = MPI.COMM_WORLD
    rank = MPI.COMM_WORLD.rank

    tscreen = terminal_screen(dt, dt*n_send, refresh_per_second=4, width = 120)
    
    # Prescribed setup
    columns = ["time", "temperature", "pressure", "velocity"]

    for i in range(n_send):

        # Generate random data row
        data = {
            "time": dt*i,  # set all 'time' values
            "temperature": np.random.rand(1)[0] ,
            "pressure_"+str(rank): np.random.rand(1)[0] ,
            "velocity": np.random.rand(1)[0] ,
        }

        if np.random.rand(1)[0]>= 0.2:
            tscreen.console.print(data)

        tscreen.update("time", data)
        
        # random action to send message
        hurdle = 0.5
        if np.random.rand(1)[0]>= hurdle:
            tscreen.update_log("Message "+str(i)+" successful. This is a random number: "+str(np.random.rand(1)[0]))
        # random action to overwrite static valuee
        if np.random.rand(1)[0]>= hurdle:
            tscreen.update("static", {"static_val1":np.random.rand(1)[0]*100 , "static_val2_"+str(rank):np.random.rand(1)[0] *5})
        
        # random delay
        time.sleep(0.5+np.random.rand(1)[0]*0.5 )

   