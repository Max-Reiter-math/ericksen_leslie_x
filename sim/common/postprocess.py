import os, warnings
import pandas as pd
import numpy as np
from pathlib import Path
from mpi4py import MPI
from dolfinx.fem import Function, functionspace, ElementMetaData
from dolfinx.io import XDMFFile, VTKFile, VTXWriter
import adios4dolfinx

class PostProcess:
    def __init__(self, comm, path: str ="", T: float = 1.0, fsr: float = 0.1, msr: float = 0.05, tbot = None, gui = None, save_as_xdmf: bool = False, save_as_checkpoint: bool = False, save_as_vtk: bool = False, save_as_vtx: bool = True):
        self.comm = comm
        self.path = path
        self. T = T                                     # end time point
        self.rounding_places = 10
        self.fsr = fsr                                  # function save rate
        self.f_savingpoints = set([0.0])
        self.msr = msr                                  # metric save rate
        self.m_savingpoints = set([0.0])
        self.GUI = gui                                  # graphical user interface
        self.save_as_xdmf = save_as_xdmf                # save functions as xdmf
        self.save_as_checkpoint = save_as_checkpoint    # checkpoint functions
        self.save_as_vtk = save_as_vtk
        self.save_as_vtx = save_as_vtx
        # 
        self.xdmf_logs = {}
        self.vtk_logs = {}
        self.vtx_logs = {}
        self.checkpoint_logs = {}

        self.static_csv = "statics.csv"
        self.temporal_csv = "time-log.csv"

        # init telegram connection and send frequency
        self.telegram_bot = tbot


    def log(self, category: str, time: str | float, data: dict, visible: bool = True):
        """
        Saves data either to a static csv or to a time
        """
        if category == "dict":
            # turn nested dictionaries into unnested ones
            normalized_data = pd.json_normalize(data, sep=".")

            if isinstance(time, (int,float)):
                rounded_time = np.round(time, self.rounding_places)
                next_saving_point = np.round(len(self.m_savingpoints)*self.msr*self.T, self.rounding_places)
                if rounded_time >= next_saving_point or rounded_time in self.m_savingpoints:   #check if this is next saving point
                    self.m_savingpoints.add(rounded_time)
                    filename = self.path + "/" + self.temporal_csv

                    normalized_data = normalized_data.set_index("time")
                
                    # check if file exists
                    if os.path.isfile(filename):
                        normalized_data.index = np.round(normalized_data.index, self.rounding_places) # rounding index as pre-requisite to merging

                        existing_df = pd.read_csv(filename, index_col="time") # reading current output
                        existing_df.index = np.round(existing_df.index, self.rounding_places)  # rounding index as pre-requisite to merging
                        resulting_df = normalized_data.combine_first(existing_df).sort_index() # merging and sorting                 
                    else:
                        resulting_df = normalized_data # if no file exists so far
                    try:
                        # print(filename)
                        resulting_df.to_csv(filename, mode = "w", encoding="utf-8")
                    except PermissionError:
                        warnings.warn("Cannot access log in csv format currently...")
                # send data to GUI
                if visible: 
                    self.GUI.update("time", data)

            elif time == "static":
                filename = self.path + "/" + self.static_csv
                # check if file exists
                if os.path.isfile(filename):
                    try:
                        normalized_data.to_csv(filename, mode = "a" , index = False)
                    except PermissionError:
                        warnings.warn("Cannot access log in csv format currently...")
                else:
                    print(filename)
                    normalized_data.to_csv(filename, mode = "w", encoding="utf-8", index = False)
                # send data to GUI
                if visible: 
                    self.GUI.update(time, data)
            # send data to telegrambot
            if self.telegram_bot!= None:
                self.telegram_bot.send_message(data)
        else:
            self.GUI.update(time, {"Warning":"No log for the parameters category "+str(category) +"and time = "+str(time)+" of type "+str(type(time)) })

    def log_functions(self, time: float, data: dict, mesh: object = None, meshtags: object = None) -> None:
        """
        Logs dolfinx functions in the via CLI pre-determined formats. Sub-functions are not accepted.
        """
        
        rounded_time = np.round(time, self.rounding_places)
        next_saving_point = np.round(len(self.f_savingpoints)*self.fsr*self.T, self.rounding_places)
        if rounded_time >= next_saving_point or rounded_time in self.f_savingpoints: 
            self.f_savingpoints.add(np.round(time, self.rounding_places)) # add new saving point
            #TODO - Generalize to instationary meshes...            
            for quantity in data.keys():
                #SECTION - SAVE TO XDMF
                if self.save_as_xdmf:            
                    # create XDMFFile if it does not exist yet
                    if quantity not in self.xdmf_logs.keys():
                        self.xdmf_logs[quantity] = XDMFFile(self.comm, self.path+"/"+str(quantity)+".xdmf", "w")
                        self.xdmf_logs[quantity].write_mesh(mesh)
                        if meshtags != None:
                            self.xdmf_logs[quantity].write_meshtags(meshtags, mesh.geometry)
                    
                    f = data[quantity]
                    try:
                        self.xdmf_logs[quantity].write_function(f, rounded_time) #, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']")
                    except RuntimeError as e:
                        try:
                            fs = f.function_space
                            dom = fs.mesh
                            element = fs.ufl_element()
                            dim = element.num_sub_elements()
                            FS = functionspace(dom,ElementMetaData("Lagrange", 1, shape = (dim,)))
                            f_tmp = Function(FS)
                            f_tmp.interpolate(f)
                            warnings.warn("Interpolating function "+str(quantity)+" into P1 space...")
                            self.xdmf_logs[quantity].write_function(f_tmp, rounded_time)
                        except Exception as e_:
                            raise RuntimeError("Encountered RuntimeError with msg "+str(e)+". Resolving the error by interpolation failed. Encountered "+ str(e_))
                #!SECTION
                #SECTION - SAVE TO VTK
                if self.save_as_vtk:
                    # create VTKFile if it does not exist yet
                    if quantity not in self.vtk_logs.keys():
                        self.vtk_logs[quantity] = VTKFile(self.comm, self.path+"/"+str(quantity)+".pvd", "w")
                        self.vtk_logs[quantity].write_mesh(mesh)
                    self.vtk_logs[quantity].write_function([data[quantity]], rounded_time ) #, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']")
                #!SECTION
                #SECTION - SAVE TO VTX
                
                if self.save_as_vtx:
                    # raise ValueError("vtx secured") # works
                    #TODO - Check if given function is the same as the in the VTKXWriter initialized function
                    # create VTXFile if it does not exist yet
                    if quantity not in self.vtx_logs.keys():
                        
                        # raise ValueError(str(self.comm.rank))
                        # if self.comm.rank == 0:
                        vtxlog = VTXWriter(self.comm, self.path+"/"+str(quantity)+".bp", data[quantity], engine="BP4")
                        # else:
                            
                            # vtxlog = None
                            # Broadcast the merged DataFrame
                        
                        # raise ValueError(str(self.comm.rank))
                        self.vtx_logs[quantity] =vtxlog # self.comm.bcast(vtxlog, root=0)
                        # mess = str(self.comm.rank) + "- log functions called" +str(time) + str(data)
                        # raise ValueError(mess)
                    # Check if it is the same function
                    self.vtx_logs[quantity].write(rounded_time)
                    # mess = str(self.comm.rank) + "- log functions called" +str(time) + str(data)
                    # raise ValueError(mess)
                #!SECTION
                #SECTION - SAVE TO CHECKPOINT
                if self.save_as_checkpoint:
                    if quantity not in self.checkpoint_logs.keys():
                        filename = Path(self.path+"/cp-"+str(quantity)+".bp")
                        self.checkpoint_logs[quantity] = filename
                        adios4dolfinx.write_mesh(self.checkpoint_logs[quantity], mesh, engine = "BP4")
                    # raise ValueError(self.checkpoint_logs[quantity], quantity)
                    adios4dolfinx.write_function(self.checkpoint_logs[quantity], data[quantity], engine = "BP4", time=rounded_time, name = quantity) 
                #!SECTION
        
    def log_message(self, message):
        self.GUI.update_log(message)
        
    def shutdown(self):
        """
        Method for irregular shutdown
        """
        # Close all xdmf files
        warnings.warn("Trying emergency shutdown for opened output files...")
        if self.telegram_bot!= None: self.telegram_bot.finishup(message = "Trying emergency shutdown for opened output files...")
        self.close()

    def close(self):
        """
        Method for regular shutdown
        """
        # Close all xdmf files
        for quantity in self.xdmf_logs:
            self.xdmf_logs[quantity].close()
        for quantity in self.vtk_logs:
            self.vtk_logs[quantity].close()
        for quantity in self.vtx_logs:
            self.vtx_logs[quantity].close()
        self.log("dict", "static",{"Status": "Shutdown succesful"})
        if self.telegram_bot!= None: self.telegram_bot.finishup(message = "Find results at "+self.path)
        