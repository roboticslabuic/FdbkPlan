import copy
import glob
import json
import os
import re
import argparse
from pathlib import Path
from datetime import datetime
import random
import subprocess
import ast
import subprocess
import shutil
from glob import glob

import openai
from google import genai
import ai2thor.controller
from ai2thor.platform import CloudRendering

import sys
sys.path.append(".")

from resources.terminal_logger import TeeLogger, InputLogger

from resources.actions import actions as available_actions
from resources import agents_properties
from resources.assess_objectType import actionable_properties
from resources.assess_objectType import context_interactions
agents = agents_properties.agents

def LM(prompt, gpt_version, max_tokens=100000, temperature=1, stop=None, logprobs=1, frequency_penalty=0):
    
    if "gpt" not in gpt_version:
        response = openai.chat.completions.create(model=gpt_version, 
                                            prompt=prompt, 
                                            max_completion_tokens=max_tokens, 
                                            temperature=temperature, 
                                            stop=stop, 
                                            logprobs=logprobs, 
                                            frequency_penalty = frequency_penalty)
        
        return response, response.choices[0].message.content
    
    else:
            
        response = openai.chat.completions.create(model=gpt_version, 
                                            messages=prompt, 
                                            max_completion_tokens=max_tokens, 
                                            temperature=temperature, 
                                            frequency_penalty = frequency_penalty)
        print("Full response from GPT model:", response)
        
        # Check for empty content (common with reasoning models hitting token limits)
        content = response.choices[0].message.content
        if not content or content.strip() == "":
            print(f"⚠️ WARNING: Empty response content!")
            print(f"   finish_reason: {response.choices[0].finish_reason}")
            print(f"   completion_tokens: {response.usage.completion_tokens}")
            if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
                print(f"   reasoning_tokens: {response.usage.completion_tokens_details.reasoning_tokens}")
            if response.choices[0].finish_reason == 'length':
                print("   → Response truncated due to token limit. The reasoning consumed all available tokens.")
        
        return response, content

def set_api_key(openai_api_key):
    openai.api_key = Path(openai_api_key + '.txt').read_text()

def get_ai2thor_objects(floor_plan_id):
    controller = ai2thor.controller.Controller(scene="FloorPlan"+str(floor_plan_id), platform=CloudRendering)
    obj = list(set([obj["objectType"] for obj in controller.last_event.metadata["objects"]]))
    controller.stop()
    return obj

def get_objs_props(obj_list):
    needed_objects_props = {}
    for obj in obj_list:
        for item in actionable_properties:
            if obj in item:
                needed_objects_props[item] = actionable_properties[item]
    return needed_objects_props

def get_objs_state(obj_list, floor_plan_id):
    controller = ai2thor.controller.Controller(scene="FloorPlan"+str(floor_plan_id), platform=CloudRendering)
    objs_state = []
    for each_obj in obj_list:
        for object in controller.last_event.metadata['objects']:
            if each_obj in object['objectType']:
                state_summary = {}
                state_summary['objectId'] = object['objectId']
                state_summary['assetId'] = object['assetId']
                state_summary['parentReceptacles'] = []
                if object['parentReceptacles']:
                    state_summary['parentReceptacles'] = [object['parentReceptacles'][0]]
                if actionable_properties[each_obj]:
                    for each_key in actionable_properties[each_obj]:
                        state_summary[each_key] = object[each_key]
                objs_state.append(state_summary)
    controller.stop()
    return objs_state

def get_objs_context(obj_list):
    objs_context = []
    for each_obj in obj_list:
        if each_obj in context_interactions:
            context_summary = {}
            context_summary['objectType'] = each_obj
            context_summary['contextual_interactions'] = context_interactions[each_obj]
            objs_context.append(context_summary)
    return objs_context

def preprocess_code(code):
    # remove markdown-style delimiters (```python and ```)
    code = re.sub(r'```[\w]*', '', code)
    code = code.strip()
    
    return code

def single_agent_code_prep(decomposed_plan_code, gpt_version):
    print (f"\n\n*******************************************************************************************************************************")
    print ("Preparing Code for Single Agent...")

    single_agent_prompt_file = open(f"./prompt_examples/floorplan_{floor_plan_type}/task_single_agent" + ".txt", "r")
    single_agent_prompt = single_agent_prompt_file.read()
    single_agent_prompt_file .close()
    prompt_code = preprocess_code(decomposed_plan_code)

    if "gpt" not in gpt_version:
        _, text = LM(prompt_code, gpt_version, max_tokens=100000, stop=["def"], frequency_penalty=0.15)
    else:            
        messages_single_agent = [{"role": "system", "content": single_agent_prompt},
                    {"role": "user", "content": prompt_code}]
        _, text = LM(messages_single_agent, gpt_version, max_tokens=100000, frequency_penalty=0.0)

    print (f"\n############################################# LLM Response For Single Agent Code #############################################################")
    print(text)

    # Ask the user for feedback
    user_input = input("\nIs this single agent code correct? (yes/no): ")
    while user_input.lower() != "yes":
        additional_instructions = input("Please provide more instructions to correct the plan: ")
        prompt_code += f"\nAdditional user instructions: {additional_instructions}"

        if "gpt" not in args.gpt_version:
            _, text = LM(prompt_code, gpt_version, max_tokens=100000, stop=["def"], frequency_penalty=0.15)
        else:
            messages_single_agent.append({"role": "assistant", "content": text})
            messages_single_agent.append({"role": "user", "content": additional_instructions})
            _, text = LM(messages_single_agent, gpt_version, max_tokens=100000, frequency_penalty=0.0)

        print(f"\n############################################# Updated LLM Response for Single Agent Code #############################################################")
        print(text)
        user_input = input("\nIs this single agent code correct now? (yes/no): ")

    # user confirms the plan is correct
    print("Single Agent Plan Code Confirmed. Proceeding...")
    single_agent_plan = preprocess_code(text)
    return single_agent_plan

def recursive_task_execution(prompt, messages, single_agent_plan, gpt_version, decomp_freq_penalty, decomposed_plan_code):
    print (f"\n\n*******************************************************************************************************************************")
    print ("Testing Task Excection...")
    executable_plan = ""
    # append the imports to the file
    start_file = Path("./resources/start_exe_test.py").read_text()
    executable_plan += (start_file + "\n")
    # append the plan code
    executable_plan += (single_agent_plan + "\n")
    # append the needed objects
    executable_plan += (needed_objects_line + "\n")
    # append the task thread termination
    end_file = Path(os.getcwd() + "/resources/end_exe.py").read_text()
    executable_plan += (end_file + "\n")

    if not os.path.isdir(f"./temp/"):
        os.makedirs(f"./temp/")
    # delete if current output already exist
    cur_path = './temp' + "/*/"
    for x in glob(cur_path, recursive = True):
        shutil.rmtree (x)

    with open("./temp/executable_plan.py", 'w') as d:
        d.write(executable_plan)

    try:
        # Run the script and capture output
        result = subprocess.run(
            ["python3", "executable_plan.py", "--floor-plan", str(args.floor_plan)],
            check=True,  # Raise CalledProcessError if script exits with non-zero status
            # stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Decode output to string
            cwd="./temp/"
        )
        print("Script ran successfully.")
        return decomposed_plan_code
    except subprocess.CalledProcessError as e:
        print(f"Script executable_plan.py failed with error: {e}")
        error_message = e.stderr
        print("Captured ", error_message)
        user_input = input("\nDo you want to fix the error? (yes/no): ")
        if user_input.lower() == "yes":
            prompt += f"\nError encountered while excecuting the code: {error_message}"
            additional_instructions = input("Any additional instructions except the error provided? ")
            prompt += f"\nAdditional user instructions: {additional_instructions}"

            if "gpt" not in gpt_version:
                _, text = LM(prompt, gpt_version, max_tokens=100000, stop=["def"], frequency_penalty=decomp_freq_penalty)
            else:
                messages.append({"role": "assistant", "content": decomposed_plan_code})
                messages.append({"role": "user", "content": additional_instructions})
                _, text = LM(messages, gpt_version, max_tokens=100000, frequency_penalty=decomp_freq_penalty) #1600 for gpt4o

            print(f"\n############################################# Updated LLM Response for Decomposed Plan Code#############################################################")
            print(text)
            # decomposed_plan_code = text

            # Ask the user for feedback
            user_input = input("\nIs this code plan correct? (yes/no): ")
            while user_input.lower() != "yes":
                additional_instructions = input("Please provide more instructions to correct the plan: ")
                prompt += f"\nAdditional user instructions: {additional_instructions}"

                if "gpt" not in gpt_version:
                    _, text = LM(prompt, gpt_version, max_tokens=100000, stop=["def"], frequency_penalty=decomp_freq_penalty)
                else:
                    messages.append({"role": "assistant", "content": text})
                    messages.append({"role": "user", "content": additional_instructions})
                    _, text = LM(messages, args.gpt_version, max_tokens=100000, frequency_penalty=decomp_freq_penalty) #1600 for gpt4o

                print(f"\n############################################# Updated LLM Response for Decomposed Plan #############################################################")
                print(text)
                user_input = input("\nIs this plan correct now? (yes/no): ")

            # user confirms the plan is correct
            print("Code Confirmed. Proceeding...")
            decomposed_plan_code = text
            
            ######################### Single Agent Code Prep #################################
            single_agent_plan = single_agent_code_prep(decomposed_plan_code, args.gpt_version)
            decomposed_plan_code = recursive_task_execution(prompt, messages, single_agent_plan, gpt_version, decomp_freq_penalty,
                                                            decomposed_plan_code)
        return decomposed_plan_code
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor-plan", type=int, required=True)
    parser.add_argument("--exp-id", type=str, required=True)
    parser.add_argument("--exp-instruction", type=str, required=True)
    parser.add_argument("--openai-api-key-file", type=str, default="api_key")
    parser.add_argument("--gpt-version", type=str, default="gpt-5", 
                        choices=['gpt-3.5-turbo' ,'gpt-4', 'gpt-4o', 'gpt-5-mini', 'gpt-5'])

    parser.add_argument("--prompt-task-desc-obj-extraction", type=str, default="task_desc_obj_extraction", 
                        choices=['task_desc_obj_extraction'])
       
    parser.add_argument("--prompt-decompse-set", type=str, default="task_decomposition", 
                        choices=['task_decomposition'])
    
    parser.add_argument("--prompt-decompse-code-set", type=str, default="task_decomposition_code", 
                        choices=['task_decomposition_code'])
    
    parser.add_argument("--prompt-allocation-set", type=str, default="task_allocation", 
                        choices=['task_allocation'])
    
    parser.add_argument("--test-set", type=str, default="final_test", 
                        choices=['final_test'])
    
    parser.add_argument("--log-results", type=bool, default=True)
    
    args = parser.parse_args()

    set_api_key(args.openai_api_key_file)

    print ("API Key Set.")

    if not os.path.isdir(f"./logs/"):
        os.makedirs(f"./logs/")
    if not os.path.isdir(f"./logs/gpt_other/"):
        os.makedirs(f"./logs/gpt_other/")
    if not os.path.isdir(f"./logs/gpt_3/"):
        os.makedirs(f"./logs/gpt_3")
    if not os.path.isdir(f"./logs/gpt_4/"):
        os.makedirs(f"./logs/gpt_4")
    if not os.path.isdir(f"./logs/gpt_5/"):
        os.makedirs(f"./logs/gpt_5")
    if not os.path.isdir(f"./logs/gpt_4o/"):
        os.makedirs(f"./logs/gpt_4o")
    if not os.path.isdir(f"./logs/gpt_5_mini/"):
        os.makedirs(f"./logs/gpt_5_mini")

    if "gpt" not in args.gpt_version:
        gpt_folder = f"gpt_other"
    elif "gpt-3.5" in args.gpt_version:
        gpt_folder = f"gpt_3"
    elif "4o" in args.gpt_version:
        gpt_folder = f"gpt_4o"
    elif "gpt-5-mini" in args.gpt_version:
        gpt_folder = f"gpt_5_mini"
    elif "gpt-5" in args.gpt_version:
        gpt_folder = f"gpt_5"
    else:
        gpt_folder = f"gpt_4"

    # prepare the example
    exp_num = args.exp_id
    test_task = args.exp_instruction

    # determine the floorplan type
    if args.floor_plan in range(1,31):
        floor_plan_type = f"kitchen"
        print('The scene is a kitchen.')
    elif args.floor_plan in range(201, 231):
        floor_plan_type = f"livingRoom"
        print('The scene is a livingRoom.')
    elif args.floor_plan in range(301, 331):
        floor_plan_type = f"bedroom"
        print('The scene is a bedroom.')
    else:
        floor_plan_type = f"bathroom"
        print('The scene is a bathroom.')

    if len(str(args.floor_plan)) == 1:
        floor_plan_num = f"00{str(args.floor_plan)}"
    elif len(str(args.floor_plan)) == 2:
        floor_plan_num = f"0{str(args.floor_plan)}"
    else:
        floor_plan_num = f"{str(args.floor_plan)}"

    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
    task_name = "{fxn}".format(fxn = '_'.join(test_task.split(' ')))
    task_name = task_name.replace('\n','')

    folder_name = f"{floor_plan_type}{floor_plan_num}_{exp_num}_{task_name}_plans_{date_time}"

    os.mkdir(f"./logs/{gpt_folder}/{folder_name}")

    terminal_log_file = open(f"./logs/{gpt_folder}/{folder_name}/terminal_log.txt", "w")
    sys.stdout = TeeLogger(terminal_log_file)
    sys.stdin = InputLogger(terminal_log_file)
    # sys.stdin = open(f"./logs/{gpt_folder}/{folder_name}/terminal_log.txt", "a")


    # change floorplan type to generic to use generic prompts.
    floor_plan_type = f"generic"
    ############################################ Extracting Objects and Locations #########################################################
    all_objects = get_ai2thor_objects(args.floor_plan)
    obj_extract_prompt_file = open(f"./prompt_examples/floorplan_{floor_plan_type}/{args.prompt_task_desc_obj_extraction}" + ".txt", "r")

    prompt = obj_extract_prompt_file.read()
    obj_extract_prompt_file.close()
    prompt += f"\n\nYou are in a different floorplan than previous, containing the following objects and locations:"
    prompt += f"\n{all_objects}"
    prompt += f"\nThe instruction: " + test_task
    print (f"\n\n******************************************************************************************************************************")
    print ("Extracting Task Description and Objects Involved...")
    print (f"\n############################################# Provided Prompt #############################################################")
    print (prompt)
    if "gpt" not in args.gpt_version:
        print ("Using non-GPT model")
        _, text = LM(prompt, args.gpt_version, max_tokens=100000, stop=["def"], frequency_penalty=0.15)
    else:
        print ("Using GPT model")
        if "4o" in args.gpt_version or "gpt-5" in args.gpt_version:
            decomp_freq_penalty = 0.0
        else:
            decomp_freq_penalty = 0.0

        messages = [{"role": "system", "content": prompt.split('\n\n')[0]},
                    {"role": "user", "content": prompt.split('\n\n')[1]},
                    {"role": "assistant", "content": prompt.split('\n\n')[2]},
                    {"role": "user", "content": prompt.split('\n\n')[3]}]
        print ("Messages prepared for GPT model.", messages)
        _, text = LM(messages,args.gpt_version, max_tokens=100000, frequency_penalty=decomp_freq_penalty)

    extracted_task_objs = text
    print (f"\n############ LLM whole Response ###############")
    print(text)
    text_lines = text.splitlines()
    needed_objects_line = next((line for line in text_lines if 'needed_objects' in line), None)
    print (f"\n############################################# LLM Response #################################################################")
    print(extracted_task_objs)
    extracted_objs_list = ast.literal_eval(needed_objects_line.split('=', 1)[1].strip())

    ################################################ Task Decomposition ####################################################################
    properties = get_objs_props(extracted_objs_list)
    states = get_objs_state(extracted_objs_list, args.floor_plan)
    context = get_objs_context(extracted_objs_list)
    decompose_prompt_file = open(f"./prompt_examples/floorplan_{floor_plan_type}/{args.prompt_decompse_set}" + ".txt", "r")
    prompt = decompose_prompt_file.read()
    decompose_prompt_file.close()
    # prompt += f"\nYou are in a floorplan containing the following objects and locations:"
    # prompt += f"\n{all_objects}\n"
    prompt += "\n\n" + extracted_task_objs
    prompt += f"\nproperties = "
    prompt += f"{properties}"
    prompt += f"\nstates = "
    prompt += f"{states}"
    prompt += f"\ncontext_interactions = "
    prompt += f"{context}"
    prompt += f"\nall_objects_available = "
    prompt += f"{all_objects}"
    prompt += f"\nIMPORTANT: Some objects are created from a source object after certain actions. For example, PotatoSliced objects are generated after a Potato object has had the Slice action used on it. Object Types that have a (*) next to them are only referenced after an interaction. For instance, Apple becomes AppleSliced once the Slice action has been applied to the Apple. Objects with a (*) in the properties dictionary don't exist in the scene but could exist after performing actions on objects existing in the scene."

    print (f"\n\n*******************************************************************************************************************************")
    print ("Generating Decompsed Plans...")
    print (f"\n############################################# Provided Prompt #############################################################")
    print(prompt)
    if "gpt" not in args.gpt_version:
        decomp_freq_penalty = 0.15
        _, text = LM(prompt, args.gpt_version, max_tokens=100000, stop=["def"], frequency_penalty=decomp_freq_penalty)
    else:
        if "4o" in args.gpt_version or "gpt-5" in args.gpt_version:
            decomp_freq_penalty = 0.0
        else:
            decomp_freq_penalty = 0.0

        messages = [{"role": "system", "content": prompt.split('\n\n')[0]},
                    {"role": "user", "content": prompt.split('\n\n')[1]},
                    {"role": "assistant", "content": prompt.split('\n\n')[2]},
                    {"role": "user", "content": prompt.split('\n\n')[3]}]
        _, text = LM(messages,args.gpt_version, max_tokens=100000, frequency_penalty=decomp_freq_penalty) #1600 for gpt4o

    # decomposed_plan = text
    print (f"\n############################################# LLM Response #############################################################")
    print(text)
    
    # Ask the user for feedback
    user_input = input("\nIs this plan correct? (yes/no): ")
    while user_input.lower() != "yes":
        additional_instructions = input("Please provide more instructions to correct the plan: ")
        prompt += f"\nAdditional user instructions: {additional_instructions}"

        if "gpt" not in args.gpt_version:
            _, text = LM(prompt, args.gpt_version, max_tokens=100000, stop=["def"], frequency_penalty=decomp_freq_penalty)
        else:
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": additional_instructions})
            _, text = LM(messages, args.gpt_version, max_tokens=100000, frequency_penalty=decomp_freq_penalty) #1600 for gpt4o

        print(f"\n############################################# Updated LLM Response #############################################################")
        print(text)
        user_input = input("\nIs this plan correct now? (yes/no): ")

    # user confirms the plan is correct
    print("Plan Confirmed. Proceeding...")
    decomposed_plan = text  
    text_lines = text.splitlines()
    needed_objects_line = next((line for line in text_lines if 'needed_objects' in line), None)
    extracted_objs_list = ast.literal_eval(needed_objects_line.split('=', 1)[1].strip())


    ################################################ Task Decomposition CODE #############################################################
    properties = get_objs_props(extracted_objs_list)
    states = get_objs_state(extracted_objs_list, args.floor_plan)
    context = get_objs_context(extracted_objs_list)
    decompose_code_prompt_file = open(f"./prompt_examples/floorplan_{floor_plan_type}/{args.prompt_decompse_code_set}" + ".txt", "r")
    prompt = decompose_code_prompt_file.read()
    decompose_code_prompt_file.close()
    prompt += decomposed_plan
    prompt += f"\nproperties = "
    prompt += f"{properties}"
    prompt += f"\nstates = "
    prompt += f"{states}"
    prompt += f"\ncontext_interactions = "
    prompt += f"{context}"
    prompt += f"\nactions = "
    prompt += f"{available_actions}"   
    prompt += f"\nall_objects_available = "
    prompt += f"{all_objects}"
    prompt += f"\nIMPORTANT: Some objects are created from a source object after certain actions. For example, PotatoSliced objects are generated after a Potato object has had the Slice action used on it. Object Types that have a (*) next to them are only referenced after an interaction. For instance, Apple becomes AppleSliced once the Slice action has been applied to the Apple. Objects with a (*) in the properties dictionary don't exist in the scene but could exist after performing actions on objects existing in the scene."
    prompt+= f"\nIMPORTANT: Do NOT include action \"CleanObject <objectId><toolObjectId><canBeUsedUpDetergentId>\" in the plan if you cannot find any cleaning tool <toolObjectId> in this scene. However, <canBeUsedUpDetergentId> is not necessary for cleaning action, use it if you can find a Detergent in the scene with the states 'canBeUsedUp'=True and 'UsedUp'=False, otherwise, <toolObjectId> would be enough."

    print (f"\n\n*******************************************************************************************************************************")
    print ("Generating Decompsed Plans CODE...")
    print (f"\n############################################# Provided Prompt #############################################################")
    print(prompt)
    if "gpt" not in args.gpt_version:
        decomp_code_freq_penalty = 0.15
        _, text = LM(prompt, args.gpt_version, max_tokens=100000, stop=["def"], frequency_penalty=decomp_code_freq_penalty)
    else:
        if "4o" in args.gpt_version or "gpt-5" in args.gpt_version:
            decomp_code_freq_penalty = 0.0
        else:
            decomp_code_freq_penalty = 0.0

        messages = []
        for i,split_prompt in enumerate(prompt.split('***')):
            if i == 0:
                messages.append({"role": "system", "content": split_prompt})
            elif i%2 == 0:
                messages.append({"role": "assistant", "content": split_prompt})
            else:
                messages.append({"role": "user", "content": split_prompt})
        _, text = LM(messages, args.gpt_version, max_tokens=100000, frequency_penalty=decomp_code_freq_penalty)   

    # decomposed_plan_code = text
    print (f"\n############################################# LLM Response for Decomposed Plan #############################################################")
    print(text)

    # Ask the user for feedback
    user_input = input("\nIs this code plan correct? (yes/no): ")
    while user_input.lower() != "yes":
        additional_instructions = input("Please provide more instructions to correct the plan: ")
        prompt += f"\nAdditional user instructions: {additional_instructions}"

        if "gpt" not in args.gpt_version:
            _, text = LM(prompt, args.gpt_version, max_tokens=100000, stop=["def"], frequency_penalty=decomp_freq_penalty)
        else:
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": additional_instructions})
            _, text = LM(messages, args.gpt_version, max_tokens=100000, frequency_penalty=decomp_freq_penalty) #1600 for gpt4o

        print(f"\n############################################# Updated LLM Response for Decomposed Plan #############################################################")
        print(text)
        user_input = input("\nIs this plan correct now? (yes/no): ")

    # user confirms the plan is correct
    print("Code Confirmed. Proceeding...")
    decomposed_plan_code = text

    ############################################## Single Agent Code Prep #################################################################
    single_agent_plan = single_agent_code_prep(decomposed_plan_code, args.gpt_version)

    ################################################# Task Execution Test #################################################################
    decomposed_plan_code = recursive_task_execution(prompt, messages, single_agent_plan, args.gpt_version, decomp_freq_penalty,
                                                    decomposed_plan_code)

    # user confirms the plan is correct
    print("Proceeding with the following Decomposed Plan Code...")
    print(decomposed_plan_code)
    # decomposed_plan_code = text

    ############################################## Task Allocation #########################################################################
    print (f"\n\n*******************************************************************************************************************************")
    print ("Allocating Tasks...")
    human_agent_affordance = agents_properties.Affordance(agents_properties.human_skills)
    robot_agent_affordance = agents_properties.Affordance(agents_properties.robot_skills)
    decomposed_plan_code = preprocess_code(decomposed_plan_code)
    actions = agents_properties.extract_actions_from_code(decomposed_plan_code)
    human_affordances = [agents_properties.calculate_action_affordance(human_agent_affordance, action) for action in actions]
    robot_affordances = [agents_properties.calculate_action_affordance(robot_agent_affordance, action) for action in actions]

    for i, action in enumerate(actions):
        action['human_affordances'] = human_affordances[i]
        action['robot_affordances'] = robot_affordances[i]

    allocate_prompt_file = open(f"./prompt_examples/floorplan_{floor_plan_type}/{args.prompt_allocation_set}" + ".txt", "r")
    allocate_prompt = allocate_prompt_file.read()
    allocate_prompt_file.close()
    prompt = decomposed_plan_code
    prompt += f"\nactions = {actions}"
    prompt += f"\nagents = {agents}"


    # print (prompt)
    if "gpt" not in args.gpt_version:
        _, text = LM(prompt, args.gpt_version, max_tokens=100000, stop=["def"], frequency_penalty=0.15)
    else:            
        messages = [{"role": "system", "content": allocate_prompt},
                    {"role": "user", "content": prompt}]
        _, text = LM(messages,args.gpt_version, max_tokens=100000, frequency_penalty=0.0) #0.30 #1800 for gpt4o

    print (f"\n############################################# LLM Response #############################################################")
    print(text)

    # Ask the user for feedback
    user_input = input("\nIs this allocated plan code correct? (yes/no): ")
    while user_input.lower() != "yes":
        additional_instructions = input("Please provide more instructions to correct the plan: ")
        prompt += f"\nAdditional user instructions: {additional_instructions}"

        if "gpt" not in args.gpt_version:
            _, text = LM(prompt, args.gpt_version, max_tokens=100000, stop=["def"], frequency_penalty=0.15)
        else:
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": additional_instructions})
            _, text = LM(messages, args.gpt_version, max_tokens=100000, frequency_penalty=0.0)

        print(f"\n############################################# Updated LLM Response #############################################################")
        print(text)
        user_input = input("\nIs this plan correct now? (yes/no): ")

    # user confirms the plan is correct
    print("Allocated Plan Code Confirmed. Proceeding...")
    allocated_plan = preprocess_code(text)

    sys.stdout = sys.stdout.terminal
    sys.stdin = sys.stdin.original_stdin
    terminal_log_file.close()

    # # determine the floorplan type
    # if args.floor_plan in range(1,31):
    #     floor_plan_type = f"kitchen"
    #     print('The scene is a kitchen.')
    # elif args.floor_plan in range(201, 231):
    #     floor_plan_type = f"livingRoom"
    #     print('The scene is a livingRoom.')
    # elif args.floor_plan in range(301, 331):
    #     floor_plan_type = f"bedroom"
    #     print('The scene is a bedroom.')
    # else:
    #     floor_plan_type = f"bathroom"
    #     print('The scene is a bathroom.')

    # if len(str(args.floor_plan)) == 1:
    #     floor_plan_num = f"00{str(args.floor_plan)}"
    # elif len(str(args.floor_plan)) == 2:
    #     floor_plan_num = f"0{str(args.floor_plan)}"
    # else:
    #     floor_plan_num = f"{str(args.floor_plan)}"
    # save generated plan
    if args.log_results:
        line = {}
        # now = datetime.now() # current date and time
        # date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
        # task_name = "{fxn}".format(fxn = '_'.join(test_task.split(' ')))
        # task_name = task_name.replace('\n','')

        # folder_name = f"{floor_plan_type}{floor_plan_num}_{exp_num}_{task_name}_plans_{date_time}"

        # os.mkdir(f"./logs/{gpt_folder}/{folder_name}")
        with open(f"./logs/{gpt_folder}/{folder_name}/log.txt", 'w') as f:
            f.write(test_task)
            f.write(f"\n\nGPT Version: {args.gpt_version}")
            f.write(f"\n\nDecomposition Frequency Penalty: {decomp_freq_penalty}")
            f.write(f"\n\nDecomposition Code Frequency Penalty: {decomp_code_freq_penalty}")
            f.write(f"\n\nFloor Plan: {args.floor_plan}")
        
        with open(f"./logs/{gpt_folder}/{folder_name}/extracted_task_desc_objs.txt", 'w') as d:
            d.write(extracted_task_objs)

        with open(f"./logs/{gpt_folder}/{folder_name}/extracted_objs_states.txt", 'w') as d:
            d.write(f"\n{get_objs_state(extracted_objs_list, args.floor_plan)}")          

        with open(f"./logs/{gpt_folder}/{folder_name}/decomposed_plan.txt", 'w') as d:
            d.write(decomposed_plan)

        with open(f"./logs/{gpt_folder}/{folder_name}/decomposed_plan.py", 'w') as d:
            d.write(decomposed_plan_code)

        with open(f"./logs/{gpt_folder}/{folder_name}/affordances.txt", 'w') as d:
            d.write(f"\n{actions}")

        with open(f"./logs/{gpt_folder}/{folder_name}/allocated_plan.py", 'w') as d:
            d.write(allocated_plan)

            