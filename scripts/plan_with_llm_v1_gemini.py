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

from google import genai
from google.genai import types
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

# Global client instance
_client = None

def get_client():
    """Get or create the global Gemini client instance."""
    global _client
    if _client is None:
        _client = genai.Client()
    return _client

def convert_to_gemini_contents(messages):
    """
    Convert our dictionary format messages to proper Gemini types.Content objects.
    
    Args:
        messages: List of messages in format [{"role": "user", "parts": ["text"]}] or simple string
        
    Returns:
        List of types.Content objects
    """
    if isinstance(messages, str):
        # Simple string prompt - wrap in proper Gemini format
        parts = [types.Part(text=messages)]
        content = types.Content(role="user", parts=parts)
        return [content]
    
    contents = []
    for message in messages:
        if isinstance(message, dict) and "role" in message and "parts" in message:
            # Convert to proper types
            parts = [types.Part(text=part) for part in message["parts"]]
            content = types.Content(role=message["role"], parts=parts)
            contents.append(content)
    return contents

def LM(prompt, model_name, max_tokens=50000, temperature=0, stop=None, logprobs=None, frequency_penalty=None, system_instruction=None):
    client = get_client()
    
    # Convert messages to proper types if needed
    contents = convert_to_gemini_contents(prompt)
    
    # Create the generation config
    config = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        stop_sequences=stop,
        system_instruction=system_instruction
    )

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config
    )
    
    return response, response.text

def set_api_key(api_key_file):
    api_key = Path(api_key_file + '.txt').read_text().strip()
    global _client
    _client = genai.Client(api_key=api_key)

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

def single_agent_code_prep(decomposed_plan_code, model_name):
    print (f"\n\n*******************************************************************************************************************************")
    print ("Preparing Code for Single Agent...")

    single_agent_prompt_file = open(f"./prompt_examples/floorplan_{floor_plan_type}/task_single_agent" + ".txt", "r")
    single_agent_prompt = single_agent_prompt_file.read()
    single_agent_prompt_file .close()
    prompt_code = preprocess_code(decomposed_plan_code)

    # Use Gemini format messages
    messages_single_agent = [
        {"role": "user", "parts": [prompt_code]}
    ]
    _, text = LM(messages_single_agent, model_name, max_tokens=50000, system_instruction=single_agent_prompt)

    print (f"\n############################################# LLM Response For Single Agent Code #############################################################")
    print(text)

    # Ask the user for feedback
    user_input = input("\nIs this single agent code correct? (yes/no): ")
    while user_input.lower() != "yes":
        additional_instructions = input("Please provide more instructions to correct the plan: ")
        
        messages_single_agent.append({"role": "model", "parts": [text]})
        messages_single_agent.append({"role": "user", "parts": [additional_instructions]})
        _, text = LM(messages_single_agent, model_name, max_tokens=50000, system_instruction=single_agent_prompt)

        print(f"\n############################################# Updated LLM Response for Single Agent Code #############################################################")
        print(text)
        user_input = input("\nIs this single agent code correct now? (yes/no): ")

    # user confirms the plan is correct
    print("Single Agent Plan Code Confirmed. Proceeding...")
    single_agent_plan = preprocess_code(text)
    return single_agent_plan

def recursive_task_execution(prompt, messages, single_agent_plan, model_name, decomposed_plan_code):
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
            additional_instructions = input("Any additional instructions except the error provided? ")
            
            messages.append({"role": "model", "parts": [decomposed_plan_code]})
            messages.append({"role": "user", "parts": [f"Error: {error_message}\nInstructions: {additional_instructions}"]})
            _, text = LM(messages, model_name, max_tokens=50000) 

            print(f"\n############################################# Updated LLM Response for Decomposed Plan Code#############################################################")
            print(text)
            # decomposed_plan_code = text

            # Ask the user for feedback
            user_input = input("\nIs this code plan correct? (yes/no): ")
            while user_input.lower() != "yes":
                additional_instructions = input("Please provide more instructions to correct the plan: ")
                
                messages.append({"role": "model", "parts": [text]})
                messages.append({"role": "user", "parts": [additional_instructions]})
                _, text = LM(messages, model_name, max_tokens=50000)

                print(f"\n############################################# Updated LLM Response for Decomposed Plan #############################################################")
                print(text)
                user_input = input("\nIs this plan correct now? (yes/no): ")

            # user confirms the plan is correct
            print("Code Confirmed. Proceeding...")
            decomposed_plan_code = text
            
            ######################### Single Agent Code Prep #################################
            single_agent_plan = single_agent_code_prep(decomposed_plan_code, args.model_name)
            decomposed_plan_code = recursive_task_execution(prompt, messages, single_agent_plan, model_name,
                                                            decomposed_plan_code)
        return decomposed_plan_code
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor-plan", type=int, required=True)
    parser.add_argument("--exp-id", type=str, required=True)
    parser.add_argument("--exp-instruction", type=str, required=True)
    parser.add_argument("--api-key-file", type=str, default="gemini_api")
    parser.add_argument("--model-name", type=str, default="gemini-2.5-pro", 
                        choices=['gemini-2.5-pro'])

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

    set_api_key(args.api_key_file)

    print ("API Key Set.")

    if not os.path.isdir(f"./logs/"):
        os.makedirs(f"./logs/")
    if not os.path.isdir(f"./logs/gemini/"):
        os.makedirs(f"./logs/gemini/")
    
    llm_folder = "gemini"

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

    folder_name = f"{floor_plan_type}{floor_plan_num}_{exp_num}_{task_name}_plan{date_time}"

    os.mkdir(f"./logs/{llm_folder}/{folder_name}")

    terminal_log_file = open(f"./logs/{llm_folder}/{folder_name}/terminal_log.txt", "w")
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

    # Use Gemini format messages
    system_prompt = prompt.split('***')[0]
    user_content = '***'.join(prompt.split('***')[1:])
    messages = [{"role": "user", "parts": [user_content]}]
    _, text = LM(messages, args.model_name, max_tokens=50000, system_instruction=system_prompt)

    extracted_task_objs = text

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

    # Use Gemini format messages
    system_prompt = prompt.split('***')[0]
    user_content = '***'.join(prompt.split('***')[1:])
    messages = [{"role": "user", "parts": [user_content]}]
    _, text = LM(messages, args.model_name, max_tokens=50000, system_instruction=system_prompt) 

    # decomposed_plan = text
    print (f"\n############################################# LLM Response #############################################################")
    print(text)
    
    # Ask the user for feedback
    user_input = input("\nIs this plan correct? (yes/no): ")
    while user_input.lower() != "yes":
        additional_instructions = input("Please provide more instructions to correct the plan: ")
        
        messages.append({"role": "model", "parts": [text]})
        messages.append({"role": "user", "parts": [additional_instructions]})
        _, text = LM(messages, args.model_name, max_tokens=50000, system_instruction=system_prompt)

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

    # Parse the complex prompt format and convert to Gemini format
    messages = []
    parts = prompt.split('***')
    system_prompt = parts[0]
    
    user_content = ""
    for i, split_prompt in enumerate(parts[1:]):
        if i % 2 == 1:  # model response
            if user_content:
                messages.append({"role": "user", "parts": [user_content]})
            messages.append({"role": "model", "parts": [split_prompt]})
            user_content = ""
        else:  # user content
            user_content += split_prompt
    
    if user_content:
        messages.append({"role": "user", "parts": [user_content]})

    _, text = LM(messages, args.model_name, max_tokens=50000, system_instruction=system_prompt)   

    # decomposed_plan_code = text
    print (f"\n############################################# LLM Response for Decomposed Plan #############################################################")
    print(text)

    # Ask the user for feedback
    user_input = input("\nIs this code plan correct? (yes/no): ")
    while user_input.lower() != "yes":
        additional_instructions = input("Please provide more instructions to correct the plan: ")
        
        messages.append({"role": "model", "parts": [text]})
        messages.append({"role": "user", "parts": [additional_instructions]})
        _, text = LM(messages, args.model_name, max_tokens=50000, system_instruction=system_prompt)

        print(f"\n############################################# Updated LLM Response for Decomposed Plan #############################################################")
        print(text)
        user_input = input("\nIs this plan correct now? (yes/no): ")

    # user confirms the plan is correct
    print("Code Confirmed. Proceeding...")
    decomposed_plan_code = text

    ############################################## Single Agent Code Prep #################################################################
    single_agent_plan = single_agent_code_prep(decomposed_plan_code, args.model_name)

    ################################################# Task Execution Test #################################################################
    decomposed_plan_code = recursive_task_execution(prompt, messages, single_agent_plan, args.model_name,
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
    messages = [{"role": "user", "parts": [prompt]}]
    _, text = LM(messages, args.model_name, max_tokens=50000, system_instruction=allocate_prompt)

    print (f"\n############################################# LLM Response #############################################################")
    print(text)

    # Ask the user for feedback
    user_input = input("\nIs this allocated plan code correct? (yes/no): ")
    while user_input.lower() != "yes":
        additional_instructions = input("Please provide more instructions to correct the plan: ")
        
        messages.append({"role": "model", "parts": [text]})
        messages.append({"role": "user", "parts": [additional_instructions]})
        _, text = LM(messages, args.model_name, max_tokens=50000, system_instruction=allocate_prompt)

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
        with open(f"./logs/{llm_folder}/{folder_name}/log.txt", 'w') as f:
            f.write(test_task)
            f.write(f"\n\nModel: {args.model_name}")
            f.write(f"\n\nFloor Plan: {args.floor_plan}")
        
        with open(f"./logs/{llm_folder}/{folder_name}/extracted_task_desc_objs.txt", 'w') as d:
            d.write(extracted_task_objs)

        with open(f"./logs/{llm_folder}/{folder_name}/extracted_objs_states.txt", 'w') as d:
            d.write(f"\n{get_objs_state(extracted_objs_list, args.floor_plan)}")          

        with open(f"./logs/{llm_folder}/{folder_name}/decomposed_plan.txt", 'w') as d:
            d.write(decomposed_plan)

        with open(f"./logs/{llm_folder}/{folder_name}/decomposed_plan.py", 'w') as d:
            d.write(decomposed_plan_code)

        with open(f"./logs/{llm_folder}/{folder_name}/affordances.txt", 'w') as d:
            d.write(f"\n{actions}")

        with open(f"./logs/{llm_folder}/{folder_name}/allocated_plan.py", 'w') as d:
            d.write(allocated_plan)

    