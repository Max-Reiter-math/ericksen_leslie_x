#!/usr/bin/env python
"""
This will be run, when the command: python -m sim_suite is called in the top-level directory.
"""
import warnings, sys
from datetime import datetime
import time
import signal
from argparse import Namespace
from sim.common import CLI
from sim.common.tgui import terminal_screen
from sim.common.postprocess import PostProcess

# Import of the numerical schemes and settings
import sim.experiments as experiments
import sim.models as models

from sim.__main__ import start_simulation_from_args

def main():
    # simulate CLI Input
    parser = CLI.get_parser()
    # DEFAULT args
    defaults = parser.parse_args([])
    args = defaults # args is of type Namespace

    # Specify general simulation settings
    # args.telegram_bot = "tbot.json"
    args.exp        = "smooth"
    args.dim        = 2
    args.vtx        = True
    args.checkpoint = True
    args.fsr        = 0.05
    args.msr        = 0.05

    args.T = 1* args.dt # for test purposes only

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
    

    #NOTE - This construction is used to cleanly shut down the still open files where data might be stored     in order to access them after e.g. a KeyboardInterrupt move of all of this into the main function    
    # MAKING SURE THAT XDMF FILES CLOSE PROPERLY BEFORE KEYBOARD INTERRUPT
    def signal_handler(sig, frame):
        clean_shutdown(elx_postprocess)
    # Register the signal handler for SIGINT (Ctrl + C)
    signal.signal(signal.SIGINT, signal_handler)
    
    

    mod_list=["LhP", "LL2P"]
    dh_list = [80,40,20,16,10]
    dt_list = [0.0005, 0.001, 0.005, 0.01, 0.05]
    # Run simulations
    for mod in mod_list:
        args.mod = mod
        for dh in dh_list:
            for dt in dt_list:
                args.dh = dh
                args.dt = dt

                args.sim_id = str(args.exp)+"_"+str(args.mod)+"_dh_"+str(args.dh)+"_dt_"+str(args.dt)
                # print(args)

                elx_postprocess = PostProcess(fsr= args.FunctionSaveRate, msr = args.MetricSaveRate, gui = TGUI, tbot = telegram_bot, save_as_xdmf = args.xdmf, save_as_vtk= args.vtk, save_as_vtx= args.vtx, save_as_checkpoint= args.checkpoint)
                
                try:
                    start_simulation_from_args(args, postprocess = elx_postprocess)
                except KeyboardInterrupt:
                    signal_handler("","")
                except EOFError:
                    signal_handler("","")
                except Exception as e:
                    print("oupsie ", str(e))


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