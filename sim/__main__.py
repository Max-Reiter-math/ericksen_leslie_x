import warnings
from datetime import datetime
import time
import signal
import os, sys
import shutil
import inspect
import logging
from argparse import Namespace

from dolfinx import log


# from sim.common.global_logging import *
from sim.common import CLI
from sim.common.tgui import terminal_screen
# from sim.lib import get_model_dict, get_exp_dict

from sim.common.postprocess import PostProcess

# Import of the numerical schemes and settings
import sim.experiments as experiments
import sim.models as models

"""
Main File
Takes arguments via command line input. See all options via:
    python -m sim -h
"""


def start_simulation_from_args(args: dict | Namespace, postprocess = None):  
    if type(args) == dict: args = Namespace(**args)

    # get unique timestamp and define ID to uniquely identify simulation run 
    timestamp = datetime.now().strftime("Y%Y-M%m-D%d_H%H-M%M-S%S")
    sim_id= "sim_"+ args.exp +"_"+ args.mod +"_dh"+str(args.dh) +"_dt"+str(args.dt)+"_"+timestamp
    if "sim_id" in vars(args).keys() and args.sim_id != "": sim_id = args.sim_id
    
    # determine location to save and make sure it exists
    output_path = args.folderpath+"output/"+sim_id
    if os.path.exists(output_path):
        print("The folder already exists. All files will be deleted before continuation. Do you want to continue? (Press ENTER to continue.)")
        answer = input()
        try:
            shutil.rmtree(output_path)
            os.makedirs(output_path)
        except PermissionError as e:
            warnings.warn("Not able to scrap existing data. Encountered Permission error "+ str(e) +". Did you leave the file "+output_path+"open?")
    else: os.makedirs(output_path)
    # set path for the postprocessing
    postprocess.path = output_path

    postprocess.log("dict", "static",{"Timestamp": timestamp, "sid": sim_id, "Status": "Starting up the terminal GUI."})
    if postprocess.telegram_bot != None: postprocess.telegram_bot.startup(message="Sid: "+sim_id)

    # obtain list of callable models and experiments from namespace of that submodule
    model_lib = {k:v for (k,v) in inspect.getmembers(models, inspect.isfunction)}
    exp_lib = {k:v for (k,v) in inspect.getmembers(experiments, inspect.isclass)}
    
    # INITIALIZE EXPERIMENT
    postprocess.log("dict", "static",{"Status": "Initalizing experiment..."})
    experiment = exp_lib[args.exp](args)
    postprocess.T = experiment.T

    # INITIALIZE MODEL WHICH STARTS THE SIMULAITON
    postprocess.log("dict", "static",{"Status": "Initalizing model..."})
    model_lib[args.mod](experiment, args,postprocess = postprocess)

    # Finished simulation
    postprocess.log("dict", "static",{"Status" : "Finished"})

def main():
    # get CLI input first
    args = CLI.get_args()

    # Initializing Postprocess
    
    #SECTION - EVTL. CONNECTION TO TELEGRAM
    # Check whether telebot has to be activated
    telegram_bot=None
    if args.telegram_bot != "": 
        try: 
            import private_only.notify_telegram as note_tele
            telegram_bot = note_tele.msg_bot(args.telegram_bot, frequency= args.TelegramUpdateRate)
        except ImportError:
            raise ImportError("This functionality is not available in the public repository.") 
    #!SECTION

        
    TGUI = terminal_screen(refresh_frequency= args.TerminalUpdateRate)
    elx_postprocess = PostProcess(fsr= args.FunctionSaveRate, msr = args.MetricSaveRate, gui = TGUI, tbot = telegram_bot, save_as_xdmf = args.xdmf, save_as_vtk= args.vtk, save_as_vtx= args.vtx, save_as_checkpoint= args.checkpoint)

    #NOTE - This construction is used to cleanly shut down the still open files where data might be stored     in order to access them after e.g. a KeyboardInterrupt move of all of this into the main function
    
    # MAKING SURE THAT XDMF FILES CLOSE PROPERLY BEFORE KEYBOARD INTERRUPT
    def signal_handler(sig, frame):
        clean_shutdown(elx_postprocess)
    # Register the signal handler for SIGINT (Ctrl + C)
    signal.signal(signal.SIGINT, signal_handler)
    
    # start simulation
    try:
        start_simulation_from_args(args, postprocess = elx_postprocess)
    except KeyboardInterrupt:
        signal_handler("","")
    except EOFError:
        signal_handler("","")


def clean_shutdown(postprocess):
    """
    This method closes all open files in the postprocess before exiting,
    e.g. in the case of a Keyboard Interrupt 
    """
    print("WARNING: Keyboard interrupt exception caught")
    time.sleep(0.1) 
    postprocess.close()
    time.sleep(0.5) 
    print("WARNING:Exiting the program.")
    sys.exit(130)


if __name__ == '__main__':
    main()    