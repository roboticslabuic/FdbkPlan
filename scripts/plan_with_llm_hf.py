import os
import re
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import ast
import subprocess
import shutil
import json

# Set transformers verbosity to reduce warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
from glob import glob

# Replace OpenAI with Hugging Face
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
import torch
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

max_tokens = 2048
bf_models = []

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        # Check if the generated tokens end with any of the stop sequences
        for stop_ids in self.stop_token_ids:
            if len(input_ids[0]) >= len(stop_ids) and torch.all(
                input_ids[0][-len(stop_ids) :] == stop_ids
            ):
                return True
        return False

# This helper class mimics the OpenAI response structure
class MockResponse:
    def __init__(self, text):
        self.choices = [type('obj', (), {'message': {'content': text}})]

class LLM_Generator:
    """
    A class to encapsulate a Hugging Face model and tokenizer for text generation.
    The model is loaded once upon initialization.
    """
    def __init__(self, model_name, quantize=True, quantization_bits=4):
        """
        Initializes the generator by loading the model and tokenizer.

        Args:
            model_name (str): The name of the Hugging Face model to load.
            quantize (bool): Whether to apply quantization.
            quantization_bits (int): The number of bits for quantization (4 or 8).
        """
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            print(f"Loading Hugging Face model: {model_name} on device: {self.device}")
            quantization_config = None
            if quantize:
                if quantization_bits == 4:
                    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
                elif quantization_bits == 8:
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()

    def generate(self, prompt_messages, max_tokens, temperature=1.0, top_p=0.95, top_k=64):
        """
        Generates text based on a list of prompt messages.

        Args:
            prompt_messages (list): A list of dictionaries in chat format (e.g., [{'role': 'user', 'content': '...'}]).
            max_tokens (int): The maximum number of new tokens to generate.
            temperature (float): The sampling temperature.
            top_k (int): The number of top-k candidates for sampling.

        Returns:
            A tuple containing (MockResponse, generated_text_string).
            Returns (None, "") on error.
        """
        if not self.model or not self.tokenizer:
            print("Model not loaded. Cannot generate text.")
            return None, ""
            
        try:
            # Apply the model's chat template to the message history
            full_prompt = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            input_ids_len = inputs.input_ids.shape[1]

            # Define the specific stop token for the model
            stop_word = "<end_of_turn>"
            stop_token_ids = self.tokenizer.encode(stop_word, add_special_tokens=False, return_tensors='pt').squeeze(0).to(self.device)
            stopping_criteria = StoppingCriteriaList([StopOnTokens([stop_token_ids])])
            
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "pad_token_id": self.tokenizer.eos_token_id,
            }

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs,
                    stopping_criteria=stopping_criteria,
                )
            
            generated_tokens = outputs[0][input_ids_len:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return MockResponse(generated_text), generated_text

        except Exception as e:
            print(f"Error generating response: {e}")
            traceback.print_exc()
            return None, ""

def parse_prompt_file(filepath):
    """
    Reads a specially formatted text file and converts it into a list of message dictionaries.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    messages = []
    # Split the content by the role markers, keeping the markers
    parts = re.split(r'(\[(?:SYSTEM|USER|ASSISTANT)\])', content)
    
    # The first part is usually empty, so we start from index 1
    for i in range(1, len(parts), 2):
        role_tag = parts[i]
        text_content = parts[i+1].strip()
        
        # Clean up the role name (e.g., '[SYSTEM]' -> 'system')
        role = role_tag.strip('[]').lower()
        
        messages.append({'role': role, 'content': text_content})
        
    return messages

def create_location_map(states_list):
    """
    Converts a detailed states list into a simple map of containers and their contents.

    Args:
        states_list (list): The list of object state dictionaries.

    Returns:
        dict: A dictionary where keys are container names and values are lists
              of the object types they contain.
    """
    # Initialize an empty dictionary to store our map.
    location_map = {}

    # Loop through every object in the provided states list.
    for obj in states_list:
        # Check if the object has a 'parentReceptacles' key and that it's not empty.
        if 'parentReceptacles' in obj and obj['parentReceptacles']:
            # Get the full ID of the container (e.g., 'Fridge|-00.44|...').
            # We assume the object is in the first listed receptacle.
            container_id = obj['parentReceptacles'][0]

            # Extract the clean container name by splitting the ID string.
            container_name = container_id.split('|')[0]
            
            # Extract the clean object type from its own ID string.
            object_type = obj['objectId'].split('|')[0]

            # If the container isn't a key in our map yet, add it with an empty list.
            location_map.setdefault(container_name, [])

            # Append the object's type to the list for its container.
            location_map[container_name].append(object_type)
            
    return location_map

def format_location_map(data):
    """
    Formats a dictionary to have each key on a new line,
    while keeping the list value on a single line.

    Args:
        data (dict): The dictionary to format.

    Returns:
        str: A nicely formatted string representation.
    """
    # Start with the opening brace
    lines = ["{"]

    # Get the list of keys to know when we're on the last one
    keys = list(data.keys())
    
    # Format each key-value pair
    for i, key in enumerate(keys):
        value = data[key]
        
        # Convert the list of items to a single-line JSON string
        value_str = json.dumps(value)
        
        # Build the line with 4-space indentation
        line = f'    "{key}": {value_str}'
        
        # Add a comma if it's not the last item
        if i < len(keys) - 1:
            line += ","
            
        lines.append(line)

    # Add the closing brace
    lines.append("}")
    
    # Join all lines with a newline character
    return "\n".join(lines)

def set_model_config(model_name):
    """
    Set Hugging Face model configuration
    For local models, this is just a placeholder
    """
    print(f"Using Hugging Face model: {model_name}")
    # You can add model-specific configurations here if needed
    return model_name

def get_ai2thor_objects(floor_plan_id):
    # Map floor_plan_id to proper AI2-THOR scene names
    if floor_plan_id == 0:
        scene_name = "FloorPlan001"  # Default to bathroom
    else:
        scene_name = f"FloorPlan{floor_plan_id}"

    print(f"Loading AI2-THOR scene: {scene_name}")
    controller = ai2thor.controller.Controller(scene=scene_name, platform=CloudRendering)
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
    # Map floor_plan_id to proper AI2-THOR scene names
    if floor_plan_id == 0:
        scene_name = "FloorPlan001"  # Default to bathroom
    else:
        scene_name = f"FloorPlan{floor_plan_id}"

    print(f"Getting object states for scene: {scene_name}")
    controller = ai2thor.controller.Controller(scene=scene_name, platform=CloudRendering)
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
    print(f"\n\n*******************************************************************************************************************************")
    print("Preparing Code for Single Agent...")

    single_agent_prompt_file = "./prompt_examples/floorplan_generic/task_single_agent.txt"
    prompt_code = preprocess_code(decomposed_plan_code)

    # For Hugging Face models, use the message format converted to string
    messages_single_agent = parse_prompt_file(single_agent_prompt_file)
    messages_single_agent.append({"role": "user", "content": "Now, apply the modifications to the following new script:\n\n" + prompt_code})
    _, text = LM.generate(messages_single_agent, max_tokens)

    print(f"\n############################################# LLM Response For Single Agent Code #############################################################")
    print(text)

    # Ask the user for feedback
    user_input = input("\nIs this single agent code correct? (yes/no): ")
    while user_input.lower() != "yes":
        additional_instructions = input("Please provide more instructions to correct the plan: ")
        prompt_code += f"\nAdditional user instructions: {additional_instructions}"

        messages_single_agent.append({"role": "assistant", "content": text})
        messages_single_agent.append({"role": "user", "content": prompt_code})
        _, text = LM.generate(messages_single_agent, max_tokens)

        print(f"\n############################################# Updated LLM Response for Single Agent Code #############################################################")
        print(text)
        user_input = input("\nIs this single agent code correct now? (yes/no): ")

    # user confirms the plan is correct
    print("Single Agent Plan Code Confirmed. Proceeding...")
    single_agent_plan = preprocess_code(text)
    return single_agent_plan

def recursive_task_execution(prompt, messages, single_agent_plan, model_name, decomposed_plan_code):
    print(f"\n\n*******************************************************************************************************************************")
    print("Testing Task Excection...")
    executable_plan = ""
    # append the imports to the file
    start_file = Path("./resources/start_exe_test.py").read_text()
    executable_plan += (start_file + "\n")
    # append the plan code
    executable_plan += (single_agent_plan + "\n")
    # append the needed objects
    if 'needed_objects_line' in globals() and needed_objects_line is not None:
        executable_plan += (needed_objects_line + "\n")
    else:
        # Fallback: create a basic needed_objects line
        executable_plan += 'needed_objects = ["Toilet", "Sink", "Mirror"]\n'
        print("Warning: Using fallback needed_objects in executable plan")
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
            # prompt += f"\nError encountered while excecuting the code: {error_message}"
            additional_instructions = input("Any additional instructions except the error provided? ")
            prompt += f"\nAdditional user instructions: {additional_instructions}"

            messages.append({"role": "assistant", "content": decomposed_plan_code})
            messages.append({"role": "user", "content": additional_instructions})
            _, text = LM.generate(messages, max_tokens)

            print(f"\n############################################# Updated LLM Response for Decomposed Plan Code#############################################################")
            print(text)
            # decomposed_plan_code = text

            # Ask the user for feedback
            user_input = input("\nIs this code plan correct? (yes/no): ")
            while user_input.lower() != "yes":
                additional_instructions = input("Please provide more instructions to correct the plan: ")
                prompt += f"\nAdditional user instructions: {additional_instructions}"

                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": additional_instructions})
                _, text = LM.generate(messages, max_tokens)

                print(f"\n############################################# Updated LLM Response for Decomposed Plan #############################################################")
                print(text)
                user_input = input("\nIs this plan correct now? (yes/no): ")

            # user confirms the plan is correct
            print("Code Confirmed. Proceeding...")
            decomposed_plan_code = text
            
            ######################### Single Agent Code Prep #################################
            single_agent_plan = single_agent_code_prep(decomposed_plan_code, args.hf_model)
            decomposed_plan_code = recursive_task_execution(
                prompt, messages, single_agent_plan, model_name, decomposed_plan_code
            )
        return decomposed_plan_code
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor-plan", type=int, required=True)
    parser.add_argument("--exp-id", type=str, required=True)
    parser.add_argument("--exp-instruction", type=str, required=True)
    parser.add_argument("--hf-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507", 
                        help="Hugging Face model name (e.g., 'Qwen/Qwen3-4B-Instruct-2507')")
    parser.add_argument("--quantize", action="store_true", default=False,
                        help="Enable 4-bit quantization to reduce memory usage (default: False)")
    parser.add_argument("--quantization-bits", type=int, default=4, choices=[4, 8],
                        help="Quantization bits (4 or 8, default: 4)")

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

    set_model_config(args.hf_model)

    print("Model Config Set.")

    if not os.path.isdir(f"./logs/"):
        os.makedirs(f"./logs/")

    # Use HF folder for Hugging Face models
    gpt_folder = f"hf_{args.hf_model.split('/')[-1]}"  # Use model name for folder

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

    os.makedirs(f"./logs/{gpt_folder}/{folder_name}", exist_ok=True)

    terminal_log_file = open(f"./logs/{gpt_folder}/{folder_name}/terminal_log.txt", "w")
    sys.stdout = TeeLogger(terminal_log_file)
    sys.stdin = InputLogger(terminal_log_file)
    # sys.stdin = open(f"./logs/{gpt_folder}/{folder_name}/terminal_log.txt", "a")

    LM = LLM_Generator(args.hf_model, quantize=args.quantize, quantization_bits=args.quantization_bits)

    # change floorplan type to generic to use generic prompts.
    floor_plan_type = f"generic"
    ############################################ Extracting Objects and Locations #########################################################
    all_objects = get_ai2thor_objects(args.floor_plan)
    obj_location_map = create_location_map(get_objs_state(all_objects, args.floor_plan))
    obj_extract_prompt_file = f"./prompt_examples/floorplan_{floor_plan_type}/{args.prompt_task_desc_obj_extraction}" + ".txt"
    messages = parse_prompt_file(obj_extract_prompt_file)
    prompt = f"You are in a new floorplan, different from any previous ones. Please create a SINGLE plan using only the following objects and locations:"
    prompt += f"\n{obj_location_map}"
    prompt += f"\nThe instruction: " + test_task

    print(f"\n\n******************************************************************************************************************************")
    print("Extracting Task Description and Objects Involved...")
    print(format_location_map(obj_location_map))

    messages.append({"role": "user", "content": prompt}) 
    _, text = LM.generate(messages, max_tokens)

    extracted_task_objs = text

    text_lines = text.splitlines()
    needed_objects_line = next((line for line in text_lines if 'needed_objects' in line), None)
    print(f"\n############################################# LLM Response #################################################################")
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
    prompt += '''
    IMPORTANT: Some objects are created from a source object after certain actions.
    For example, a Potato becomes PotatoSliced after the Slice action.
    Object Types marked with (*) only appear after an interaction—for instance, an Apple becomes AppleSliced after slicing.
    Objects with a (*) in their properties dictionary are not present in the scene initially but can be created after performing actions on existing objects.
    However, toasting does not create a new object type.
    For example, Bread remains Bread after being toasted; it does not become BreadToasted*.
    Additionally, note that both PutObject and DropObject functions already include PickUpObject and GoToObject.
    Therefore, DO NOT pick up, get, or go to the object.
    '''

    print(f"\n\n*******************************************************************************************************************************")
    print("Generating Decomposed Plans...")
    # print(f"\n############################################# Provided Prompt #############################################################")
    # print(prompt)

    messages = [{"role": "system", "content": prompt.split('\n\n')[0]},
                {"role": "user", "content": prompt.split('\n\n')[1]},
                {"role": "assistant", "content": prompt.split('\n\n')[2]},
                {"role": "user", "content": prompt.split('\n\n')[3]}]
    _, text = LM.generate(messages, max_tokens)

    # decomposed_plan = text
    print(f"\n############################################# LLM Response #############################################################")
    print(text)
    
    # Ask the user for feedback
    user_input = input("\nIs this plan correct? (yes/no): ")
    while user_input.lower() != "yes":
        additional_instructions = input("Please provide more instructions to correct the plan: ")
        prompt += f"\nAdditional user instructions: {additional_instructions}"

        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": additional_instructions})
        _, text = LM.generate(messages, max_tokens)

        print(f"\n############################################# Updated LLM Response #############################################################")
        print(text)
        user_input = input("\nIs this plan correct now? (yes/no): ")

    # user confirms the plan is correct
    print("Plan Confirmed. Proceeding...")
    decomposed_plan = text
    text_lines = text.splitlines()
    needed_objects_line = next((line for line in text_lines if 'needed_objects' in line), None)

    if needed_objects_line is None:
        print("Warning: Could not find 'needed_objects' in decomposed plan. Using previous objects.")
        # Keep the previously extracted objects
        pass
    else:
        try:
            extracted_objs_list = ast.literal_eval(needed_objects_line.split('=', 1)[1].strip())
        except (ValueError, SyntaxError) as e:
            print(f"Warning: Could not parse needed_objects from decomposed plan: {e}")
            # Keep the previously extracted objects
            pass


    ################################################ Task Decomposition CODE #############################################################
    properties = get_objs_props(extracted_objs_list)
    states = get_objs_state(extracted_objs_list, args.floor_plan)
    context = get_objs_context(extracted_objs_list)
    messages = parse_prompt_file(f"./prompt_examples/floorplan_{floor_plan_type}/{args.prompt_decompse_code_set}" + ".txt")
    prompt = decomposed_plan
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

    print(f"\n\n*******************************************************************************************************************************")
    print("Generating Decomposed Plans CODE...")
    # print(f"\n############################################# Provided Prompt #############################################################")
    # print(prompt)

    messages.append({"role": "user", "content": prompt})
    _, text = LM.generate(messages, max_tokens)

    # decomposed_plan_code = text
    print(f"\n############################################# LLM Response for Decomposed Plan #############################################################")
    print(text)

    # Ask the user for feedback
    user_input = input("\nIs this code plan correct? (yes/no): ")
    while user_input.lower() != "yes":
        additional_instructions = input("Please provide more instructions to correct the plan: ")
        prompt += f"\nAdditional user instructions: {additional_instructions}"

        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": additional_instructions})
        _, text = LM.generate(messages, max_tokens)

        print(f"\n############################################# Updated LLM Response for Decomposed Plan #############################################################")
        print(text)
        user_input = input("\nIs this plan correct now? (yes/no): ")

    # user confirms the plan is correct
    print("Code Confirmed. Proceeding...")
    decomposed_plan_code = text

    ############################################## Single Agent Code Prep #################################################################
    single_agent_plan = single_agent_code_prep(decomposed_plan_code, args.hf_model)

    ################################################# Task Execution Test #################################################################
    decomposed_plan_code = recursive_task_execution(prompt, messages, single_agent_plan, args.hf_model, decomposed_plan_code)

    # user confirms the plan is correct
    print("Proceeding with the following Decomposed Plan Code...")
    print(decomposed_plan_code)
    # decomposed_plan_code = text

    ############################################## Task Allocation #########################################################################
    print(f"\n\n*******************************************************************************************************************************")
    print("Allocating Tasks...")
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


    # print(prompt)         
    messages = [{"role": "system", "content": allocate_prompt},
                {"role": "user", "content": prompt}]
    _, text = LM.generate(messages, max_tokens) #0.30 #1800 for gpt4o

    print(f"\n############################################# LLM Response #############################################################")
    print(text)

    # Ask the user for feedback
    user_input = input("\nIs this allocated plan code correct? (yes/no): ")
    while user_input.lower() != "yes":
        additional_instructions = input("Please provide more instructions to correct the plan: ")
        prompt += f"\nAdditional user instructions: {additional_instructions}"

        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": additional_instructions})
        _, text = LM.generate(messages, max_tokens)

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
            f.write(f"\n\nModel: {args.hf_model}")
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
