# import time, sys, os
# import threading as threading

from datetime import datetime
import os, sys

# TODO make it object-oriented + refreshing frequency
# make little test if warnings can be logged 
# otherwise stay with current methodology

# Initialize the screen



# Check if screen was re-sized (True or False)


class terminal_screen:
    def __init__(self, refresh_frequency = 5):
        self.starting_time = datetime.now()
        self.last_refreshed = datetime.now()
        self.last_progressed = datetime.now()
        self.refresh_frequenecy = refresh_frequency # in seconds
        self.content = {} # always as a dictionary
        self.last_progress = 0.0
        self.avg_progression_time = 0.0 # time predictor for 1% progression
        

    def update(self, new_dict):
        # update values
        for key in new_dict.keys():
            self.content[key] = new_dict[key]
        # visualize
        self.refresh()

    def refresh(self):
        cols, lines = os.get_terminal_size()
        # only refresh in the initalized frequency
        if (datetime.now() - self.last_refreshed).total_seconds() >= self.refresh_frequenecy:
            # print upper border
            sys.stdout.write(border(cols))
            # print given values
            sys.stdout.write(dict_to_str(self.content, exceptions = ["progress"]))

            # progress bar and time prediction
            if "progress" in self.content.keys():
                progress = self.content["progress"]
                # print progress bar
                sys.stdout.write("_"*cols + "\n")
                sys.stdout.write(progressbar(self.content["progress"], cols))
                elapsed_time =  (datetime.now() - self.starting_time).total_seconds()
                # if progress is made update weighted average of elapsed time per progress point  
                if progress > self.last_progress:                    
                    time_since_last_progress = (datetime.now() - self.last_progressed).total_seconds()
                    weight = 0.5
                    if self.avg_progression_time > 0.0:
                        self.avg_progression_time = ((1-weight)* self.avg_progression_time + weight * time_since_last_progress/(progress-self.last_progress) )
                    else:
                        self.avg_progression_time = time_since_last_progress/(progress-self.last_progress)
                    self.last_progressed = datetime.now()
                # predict remaining time         
                predicted_time =  self.avg_progression_time*(1-progress)
                sys.stdout.write("Elapsed time: "+ sec_to_dhms(elapsed_time) +"\nEstimated remaining time: " + sec_to_dhms(predicted_time)+"\n")
                # update heuristics
                self.last_progress = progress  

            self.last_refreshed = datetime.now()
            sys.stdout.write("Last updated: "+self.last_refreshed.strftime("%m/%d/%Y, %H:%M:%S") + "\n")
            sys.stdout.flush()
            


def border(cols):
    return "#"*cols + "\n"*4 + "#"*cols + "\n"

def dict_to_str(dic, exceptions = []):
    res = ""
    for key in dic.keys():
        if key not in exceptions:
            res += str(key) + " : "+ str(dic[key]) +"\n"
    return res

def progressbar(progress, cols):
    bar_length = max(10, int(cols/2))
    x = int(progress * bar_length )
    return "Progress : [" + "■"*x +"·"*(bar_length-x) + "] "+ str(int(progress*100)) + " % " +"\n"

def sec_to_dhms(sec):
    """
    replacement for .strftime since timeit.default_timer does not have that option
    """
    d = sec // (3600*24)
    h = sec // 3600
    m = (sec-h*3600) // 60
    sec = int(sec-m*60-h*3600)
    res = ""
    if d >0: res += str(d) + " days " 
    if h >0: res += str(h)+" hours "
    if m >0: res += str(m)+" minutes "
    res += str(sec) + " seconds"
    return res    

def test():
    import time
    import random
    tui = terminal_screen()
    for i in range(20):
        # print(i)
        time.sleep(1)
        for line in range(20-i):
            time.sleep(.25)
            data = {str(random.randint(0,15)): random.randint(0,100), "progress" : i/20}
            # print(dict_to_str(data))
            tui.update(data)
    tui.close()
        
