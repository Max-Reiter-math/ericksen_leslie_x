
import os
import itertools
import json
import argparse

"""
Creates bash files to execute based on the json files in this folder.
Creates the combinations of all given parameters in the json files and automatically enumerates the simulations.

python -m sim.sim_presets.create -end "|| true"
"""

def create_bash_from_dict(filename: str, config_json: dict, mpirun = "mpirun", end = "&&"):
    param_combinations = list(itertools.product(*config_json.values()))
    param_combinations = list(map(list, param_combinations))
    keys = list(config_json.keys())

    with open ('sim/sim_presets/'+filename+'.sh', 'w') as f:
        f.write('')

    for j in range(len(param_combinations)):
        params = param_combinations[j]
        command = ""
        indices = list(range(len(keys)))

        if "-n" in keys:
            core_index = keys.index("-n")
            indices.remove(core_index)
            if params[core_index]>1:
                command += mpirun + " -n "+str(params[core_index])+" "

        if "-sid" in keys:
            sid_index = keys.index("-sid")
            params[sid_index] += str(j+1)


        command += "python -m sim.run"
        for i in indices:
            command += " "
            command += str(keys[i])
            command += " "
            command += str(params[i])

        if (j+1) != len(param_combinations):
            command += " " + end 
            # && --> STOP IMMEDIATELY IF COMMAND FAILS
            # || true KEEP GOING IF COMMAND FAILS

        with open ('sim/sim_presets/'+filename+'.sh', 'a') as f:
            f.write(command)
            f.write("\n")
        


    print("Succesfully written bash file to location:", 'sim/sim_presets/'+filename+'.sh')


def get_all_jsons():
    path_to_json = 'sim/sim_presets/'
    json_files = [(path_to_json + pos_json) for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    print("Available JSON Files:", json_files)
    return json_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI for the creation of automated bash files.")    
    parser.add_argument('-mpirun', type=str, default = "mpirun", help='Command/Path to call mpirun. Default: mpirun')
    parser.add_argument('-end', type=str, default = "&&", help='Appendix to end of line. Default: &&')
    args = parser.parse_args()

    # find all json files
    json_files = get_all_jsons()

    for json_file in json_files:   
        # read json file   
        with open(json_file, 'r') as file:
            json_dict = json.load(file)

        # write to bash file
        filename = json_file.split("/")[-1].replace(".json","")
        create_bash_from_dict(filename, json_dict, mpirun = args.mpirun, end = args.end)
        

        
            