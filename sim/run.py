import warnings
import time
import signal
import os, sys
import shutil
import json
from mpi4py import MPI

from sim.io import cli
from sim.io.tgui import terminal_screen
from sim.common.postprocess import PostProcess

"""
Main File
Takes arguments via command line input. See all options via:
    python -m sim.run -h
    mpirun -n 3 python -m sim.run -h
"""

def mkdir(args):
    # determine location to save and make sure it exists
    output_path = args.folderpath+"output/"+args.sim_id
    if os.path.exists(output_path):
        print("The folder already exists. All files will be deleted before continuation.")
        if args.overwrite == 0:
            print("Do you want to continue? (Press ENTER to continue.)")
            answer = input()
        else:
            print("Results will be overwritten as given by the cli input...")
            
        try:
            shutil.rmtree(output_path)
            os.makedirs(output_path)
        except PermissionError as e:
            warnings.warn("Not able to scrap existing data. Encountered Permission error "+ str(e) +". Did you leave the file "+output_path+"open?")
    else: os.makedirs(output_path)

def save_config(args):
    # determine location to save and make sure it exists
    output_path = args.folderpath+"output/"+args.sim_id
    with open(output_path+"/config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

def main():
    # Start with collecting the rank communication
    comm = MPI.COMM_WORLD

    # we initialize the args only on rank 0 and then share it across all ranks
    if comm.rank == 0:    
        args, mod, exp = cli.parse_args()   # get arguments from cli input
        mkdir(args)                         # create output folder
        save_config(args)                   # save config json

    else:
        args, mod, exp = None, None, None    

    # Broadcast args, mod and exp to all ranks
    args = comm.bcast(args, root=0)  
    exp = comm.bcast(exp, root=0)  
    mod = comm.bcast(mod, root=0) 

    start_sim(comm,args,exp,mod)


def start_sim(comm,args,exp,mod):
    # Create GUI in the terminal and a postprocess to log everything.
    # Both are created once for every rank!
    # This is not an issue since the VTXWriter, XDMFWriter class etc. all are able to communicate via MPI.COMM_WORLD.
    # Further, the TGUI is only manually refreshed on an update in a multi-rank mpirun.
    TGUI = terminal_screen(dt=args.dt, T= args.T, width = args.twidth, height = args.theight)
    elx_postprocess = PostProcess(comm, path = "output/"+args.sim_id, fsr= args.functionsaverate, msr = args.metricsaverate, gui = TGUI, tbot = None, save_as_xdmf = args.xdmf, save_as_vtk= args.vtk, save_as_vtx= args.vtx, save_as_checkpoint= args.checkpoint)
    # elx_postprocess = None # For testing only

    
    experiment = exp(comm,args) # initialize experiment
    
    # MAKING SURE THAT SAVED FILES CLOSE PROPERLY BEFORE KEYBOARD INTERRUPT
    def signal_handler(sig, frame):
        clean_shutdown(elx_postprocess)
    # Register the signal handler for SIGINT (Ctrl + C)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:        
        model = mod(comm, experiment, args,postprocess = elx_postprocess) # simulation start
    except KeyboardInterrupt:
        signal_handler("","")
    except EOFError:
        signal_handler("","")


def clean_shutdown(postprocess):
    """
    This method closes all open files in the postprocess before exiting,
    e.g. in the case of a Keyboard Interrupt 
    """
    if MPI.COMM_WORLD.rank==0:
        print("WARNING: Keyboard interrupt exception caught")
        time.sleep(0.1) 
        postprocess.close()
        time.sleep(0.5) 
        print("WARNING:Exiting the program.")
        sys.exit(130)


if __name__ == '__main__':
    main()    