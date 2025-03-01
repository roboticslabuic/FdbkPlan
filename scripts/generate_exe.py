import os
from pathlib import Path
import subprocess
import argparse

def compile_aithor_exec_file(gpt_dir, expt_name):
    log_path = os.getcwd() + "/logs/" + gpt_dir + "/" + expt_name
    #print(log_path)
    executable_plan = ""
    
    # append the imports to the file
    start_file = Path(os.getcwd() + "/resources/start_exe.py").read_text()
    executable_plan += (start_file + "\n")
    
    # append the allocated plan
    allocated_plan = Path(log_path + "/allocated_plan.py").read_text()
    executable_plan += (allocated_plan + "\n")

    # append the needed objects
    text = Path(log_path + "/decomposed_plan.txt").read_text()
    text_lines = text.splitlines()
    needed_objects_line = next((line for line in text_lines if 'needed_objects' in line), None)
    # executable_plan += (text.split("\n")[-1] + "\n")
    executable_plan += (needed_objects_line + "\n")

    # append the task thread termination
    end_file = Path(os.getcwd() + "/resources/end_exe.py").read_text()
    executable_plan += (end_file + "\n")

    with open(f"{log_path}/executable_plan.py", 'w') as d:
        d.write(executable_plan)
        
    return (f"{log_path}/executable_plan.py")

parser = argparse.ArgumentParser()
parser.add_argument("--gpt", type=str, default="gpt_4o")
parser.add_argument("--exp", type=str, required=True)
args = parser.parse_args()

gpt_dir = args.gpt
expt_name = args.exp
print (gpt_dir, expt_name)
ai_exec_file = compile_aithor_exec_file(gpt_dir, expt_name)

# subprocess.run(["python3", ai_exec_file])
