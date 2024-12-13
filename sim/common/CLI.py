from argparse import ArgumentParser, Namespace
import inspect
import sim.models as models
import sim.experiments as experiments

def get_parser():
    # obtain list of callable models and experiments from namespace of that submodule
    model_lib = [k for (k,v) in inspect.getmembers(models, inspect.isfunction)]
    exp_lib = [k for (k,v) in inspect.getmembers(experiments, inspect.isclass)]
    
    #SECTION - Define Arguement Parser
    parser = ArgumentParser(description="This program runs a single simulation with specification by command line input.")

    # arguments given via command line
    # Model and experiment arguments
    megroup = parser.add_argument_group("Model and experiment settings")
    megroup.add_argument('-m','--mod', type=str, metavar='', help='THIS ARGUMENT IS REQUIRED: The model considered in the simulation. Options are ' + ", ".join(model_lib)) 
    megroup.add_argument('-s','--submod', type=int, metavar='', nargs='?', const=1, default=1, help='The ericksen-leslie model considered in the simulation. The choices depend on the model. Default is 1.')
    megroup.add_argument('-e','--exp', type=str, metavar='', help='THIS ARGUMENT IS REQUIRED: The experiment considered in the simulation. Options are '+ ", ".join(exp_lib))
    megroup.add_argument('-d','--dim', type=int, metavar='', help='Configures the dimension of the experiment, if dim is supported by the chosen experiment')
    megroup.add_argument('-dt','--dt', type=float, metavar='', help='Specifies resolution of time partition. Default is given by the experiment.')
    megroup.add_argument('-dh','--dh', type=int, metavar='', help='Specifies resolution of space partition. Default is given by the experiment.')
    megroup.add_argument('-T','--T', type=float, metavar='', help='Specifies resolution of time partition. Default is given by the experiment.')
    megroup.add_argument('-a','--alpha', type=float, metavar='', nargs='?', const=0.1, default=0.1, help='Specifies the regularization parameter. This is alpha for the dg Method or e.g. epsilon for penalizaton method. Default is given by 0.1.')

    
    
    # Postprocessing formats (one needs to be selected)
    outputgroup = parser.add_argument_group("Output settings")
    outputgroup.add_argument('-fp', '--folderpath', type=str, metavar='', nargs='?', const="", default="", help='Define the path to the folder for the output of the simulations.')
    outputgroup.add_argument('-sid', '--sim_id', type=str, metavar='', nargs='?', const="", default="", help='Overwrite the name of the simulation folder. Otherwise a unique identifier is chosen.')
    outputgroup.add_argument('-xdmf','--xdmf', action="store_true", help='Store FEM functions in xdmf format. XDMF Files only work with Lagrange spaces of first order.')
    outputgroup.add_argument('-vtx','--vtx', action="store_true", help='Store FEM functions in vtx (.bp) format. VTX Files are very flexible with respect to the FEM space.')
    outputgroup.add_argument('-vtk','--vtk', action="store_true", help='Store FEM functions in vtk (.pvd) format.  VTK Files only work with Lagrange spaces of  order <= 2 or DG1 spaces.')
    outputgroup.add_argument('-cp','--checkpoint', action="store_true", help='Checkpoint FEM functions to compare later.')
    outputgroup.add_argument('-fsr','--FunctionSaveRate', type=float, metavar='', nargs='?', const=0.1, default=0.1, help='Save frequency of functions. The input 0.1 saves the function every 10 percent of the temporal evoution --> 11 saving points in time.')
    outputgroup.add_argument('-msr','--MetricSaveRate', type=float, metavar='', nargs='?', const=0.05, default=0.05, help='Save frequency of function metrics such as Energy. The input 0.1 saves the function every 10 percent of the temporal evoution --> 11 saving points in time.')

    # default arguments
    uxgroup = parser.add_argument_group("User interface settings")
    uxgroup.add_argument('-tur','--TerminalUpdateRate', type=float, metavar='', nargs='?', const=5.0, default=5.0, help='Update rate of metrics in the terminal in seconds.')
    uxgroup.add_argument('-tbot', '--telegram_bot', type=str, metavar='', nargs='?', const="", default="", help='Path to the configuration json for a telegram bot. You can find an example in the file telegram_example.json .')
    uxgroup.add_argument('-tbotur','--TelegramUpdateRate', type=int, metavar='', nargs='?', const=1800, default=1800, help='Maximal update rate of metrics via telegram in seconds. Default is half an hour.')

    return parser

def get_args():
    """
    This method collects all arguments from the command line input. T
    To see the options call:
    python -m sim -h
    """
    parser = get_parser()

    # parse arguments
    args = parser.parse_args()
    #!SECTION

    #SECTION - Sanity check of parsed arguments

    # Check if the at least one saving format is activated
    if args.xdmf == False and args.vtk == False and args.vtx == False and args.checkpoint == False:
        print("You have selected no format for saving the function. No function data will be saved during the simulation. Check python -m sim -h for options. Do you want to continue? (Press ENTER to continue.)")
        answer = input()

    # Check if the most necessary arguments are provided for the argument call
    if args.mod == None or args.exp == None:
        raise TypeError("You must specify the model and the experiment to run a simulation either via command line or via a config file. Type \'python sim.py -h\' for help.")
    
    #!SECTION

    return args



    