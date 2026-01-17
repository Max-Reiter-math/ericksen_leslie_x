import time
import argparse
import sys
import json
from mpi4py import MPI

from sim.io.tgui import terminal_screen
from sim.common.postprocess import PostProcess
from sim.io.cli import MODEL_REGISTRY, EXPERIMENT_REGISTRY
from sim.run import start_sim


"""
Run File from existing config file, in particular to RERUN a simulation.
    python -m sim.runconfig "output/sim_id123/config.json"
    mpirun -n 3 python -m sim.runconfig "output/sim_id123/config.json"
"""

def read_config(filepath: str) -> argparse.Namespace:
    with open(filepath, "r") as f:
        config = json.load(f)
    return argparse.Namespace(**config)

def main():
    # Start with collecting the rank communication
    comm = MPI.COMM_WORLD

    # we initialize the args only on rank 0 and then share it across all ranks
    if comm.rank == 0:
        # get config.json path via custom cli
        parser = argparse.ArgumentParser(description="Run Simulation from Config File")
        parser.add_argument("filepath", type=str, help="Path to config file, e.g. outputs/testimsulation/config.json")
        fargs = parser.parse_args()
        
        # load args from config.json
        args = read_config(fargs.filepath)
        mod = MODEL_REGISTRY[args.mod]
        exp = EXPERIMENT_REGISTRY[args.exp]
        
    else:
        args, mod, exp = None, None, None    

    # Broadcast args, mod and exp to all ranks
    args = comm.bcast(args, root=0)  
    exp = comm.bcast(exp, root=0)  
    mod = comm.bcast(mod, root=0) 
    
    start_sim(comm,args,exp,mod)

if __name__ == '__main__':
    main()    