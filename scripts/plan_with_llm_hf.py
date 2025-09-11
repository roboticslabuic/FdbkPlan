import glob
import os
import re
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import ast
import subprocess
import shutil

# Set transformers verbosity to reduce warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
from glob import glob

# Replace OpenAI with Hugging Face
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
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

max_tokens = 1024
decomp_freq_penalty = 0.0
decomp_code_freq_penalty = 0.0
bf_models = []

def LM(prompt, model_name, max_tokens, temperature=0, stop=None, logprobs=True, frequency_penalty=0, top_p=None, top_k=None, quantize=True, quantization_bits=4):
    """
    Hugging Face model inference function with quantization support
    
    Parameters:
    - frequency_penalty: Mapped to repetition_penalty in HuggingFace (positive values penalize repetition)
    - logprobs: When True, enables output_scores=True and return_dict_in_generate=True
    - temperature, top_p, top_k: Standard sampling parameters
    - stop: Post-processed after generation (HF pipeline doesn't support stop sequences directly)
    """
    # Initialize model and tokenizer if not already done
    if not hasattr(LM, 'model'):
        print(f"Loading Hugging Face model: {model_name}")
        try:
            # Configure quantization if enabled
            quantization_config = None
            if quantize:
                try:
                    print(f"Using {quantization_bits}-bit quantization")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=(quantization_bits == 4),
                        load_in_8bit=(quantization_bits == 8),
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                except Exception as quant_error:
                    print(f"Warning: Could not configure quantization: {quant_error}")
                    print("Falling back to full precision mode")
                    quantization_config = None
                    quantize = False

            LM.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Handle missing pad_token for some models
            if LM.tokenizer.pad_token is None:
                LM.tokenizer.pad_token = LM.tokenizer.eos_token

            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            LM.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config if quantize else None,
                dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )

            if device == "cpu":
                LM.model = LM.model.to(device)

            LM.pipe = pipeline(
                "text-generation",
                model=LM.model,
                tokenizer=LM.tokenizer,
                dtype="auto",
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, ""

    # Convert messages format to string if needed
    if isinstance(prompt, list):
        # Handle OpenAI-style messages
        full_prompt = ""
        for message in prompt:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "system":
                full_prompt += f"System: {content}\n"
            elif role == "user":
                full_prompt += f"User: {content}\n"
            elif role == "assistant":
                full_prompt += f"Assistant: {content}\n"
        full_prompt += "Assistant: "
    else:
        full_prompt = prompt

    # Generate response
    try:
        # Prepare generation parameters that are supported by transformers
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": LM.tokenizer.eos_token_id,
            "return_full_text": False
        }
        
        # Enable sampling if any sampling parameters are provided
        should_sample = temperature > 0 or (top_p is not None and top_p < 1.0) or (top_k is not None and top_k > 0)
        generation_kwargs["do_sample"] = should_sample
        
        # Only add sampling parameters if sampling is enabled
        if should_sample:
            if temperature > 0:
                generation_kwargs["temperature"] = temperature
            else:
                generation_kwargs["temperature"] = 0.01  # Small non-zero value for sampling
                
            if top_p is not None and top_p > 0:
                generation_kwargs["top_p"] = top_p
                
            if top_k is not None and top_k > 0:
                generation_kwargs["top_k"] = top_k
        
        # Map frequency_penalty to repetition_penalty in HuggingFace
        if frequency_penalty != 0:
            # Convert frequency_penalty to repetition_penalty
            # frequency_penalty is typically negative, repetition_penalty is > 1.0 for penalizing repetition
            if frequency_penalty > 0:
                generation_kwargs["repetition_penalty"] = 1.0 + frequency_penalty
            else:
                # For negative frequency_penalty, we encourage repetition (< 1.0)
                generation_kwargs["repetition_penalty"] = max(0.1, 1.0 + frequency_penalty)
        
        # Handle logprobs - Note: pipeline doesn't directly support output_scores
        # We'll need to use the model directly for logprobs functionality
        use_raw_model = logprobs
        
        if use_raw_model:
            # When logprobs are requested, use model.generate() directly instead of pipeline
            generation_kwargs["output_scores"] = True
            generation_kwargs["return_dict_in_generate"] = True
        
        # Filter out any unsupported parameters before passing to pipeline
        if use_raw_model:
            # Parameters for model.generate()
            supported_params = {
                'max_new_tokens', 'temperature', 'top_p', 'top_k', 'do_sample', 
                'pad_token_id', 'num_return_sequences', 'repetition_penalty', 
                'length_penalty', 'early_stopping', 'output_scores', 'return_dict_in_generate'
            }
        else:
            # Parameters for pipeline
            supported_params = {
                'max_new_tokens', 'temperature', 'top_p', 'top_k', 'do_sample', 
                'pad_token_id', 'return_full_text', 'num_return_sequences',
                'repetition_penalty', 'length_penalty', 'early_stopping'
            }
        
        filtered_kwargs = {k: v for k, v in generation_kwargs.items() if k in supported_params}
        
        if use_raw_model:
            # Use model.generate() directly for logprobs support
            inputs = LM.tokenizer(full_prompt, return_tensors="pt").to(LM.model.device)
            with torch.no_grad():
                outputs = LM.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **filtered_kwargs
                )
            
            # Handle different output formats based on return_dict_in_generate
            if hasattr(outputs, 'sequences'):
                # When return_dict_in_generate=True, outputs is a GenerateOutput object
                generated_tokens = outputs.sequences[0][len(inputs.input_ids[0]):]
            else:
                # When return_dict_in_generate=False, outputs is just the token sequences
                generated_tokens = outputs[0][len(inputs.input_ids[0]):]
            
            generated_text = LM.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            # Use pipeline for regular generation
            outputs = LM.pipe(full_prompt, **filtered_kwargs)
            generated_text = outputs[0]['generated_text']
        
        # Post-process to handle stop sequences if provided
        if stop and generated_text:
            stop_list = stop if isinstance(stop, list) else [stop]
            for stop_word in stop_list:
                if stop_word in generated_text:
                    generated_text = generated_text.split(stop_word)[0]
                    break

        # Create a mock response object similar to OpenAI's format
        class MockResponse:
            def __init__(self, text):
                self.choices = [type('obj', (object,), {'message': type('obj', (object,), {'content': text})()})()]

        return MockResponse(generated_text), generated_text

    except Exception as e:
        print(f"Error generating response: {e}")
        return None, ""

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
    print (f"\n\n*******************************************************************************************************************************")
    print ("Preparing Code for Single Agent...")

    single_agent_prompt_file = open(f"./prompt_examples/floorplan_generic/task_single_agent" + ".txt", "r")
    single_agent_prompt = single_agent_prompt_file.read()
    single_agent_prompt_file .close()
    prompt_code = preprocess_code(decomposed_plan_code)

    # For Hugging Face models, use the message format converted to string
    messages_single_agent = [
        {"role": "system", "content": single_agent_prompt},
        {"role": "user", "content": prompt_code}
    ]
    _, text = LM(messages_single_agent, model_name, max_tokens, frequency_penalty=0.0, quantize=args.quantize, quantization_bits=args.quantization_bits)

    print (f"\n############################################# LLM Response For Single Agent Code #############################################################")
    print(text)

    # Ask the user for feedback
    user_input = input("\nIs this single agent code correct? (yes/no): ")
    while user_input.lower() != "yes":
        additional_instructions = input("Please provide more instructions to correct the plan: ")
        prompt_code += f"\nAdditional user instructions: {additional_instructions}"

        messages_single_agent.append({"role": "assistant", "content": text})
        messages_single_agent.append({"role": "user", "content": additional_instructions})
        _, text = LM(messages_single_agent, model_name, max_tokens, frequency_penalty=0.0, quantize=args.quantize, quantization_bits=args.quantization_bits)

        print(f"\n############################################# Updated LLM Response for Single Agent Code #############################################################")
        print(text)
        user_input = input("\nIs this single agent code correct now? (yes/no): ")

    # user confirms the plan is correct
    print("Single Agent Plan Code Confirmed. Proceeding...")
    single_agent_plan = preprocess_code(text)
    return single_agent_plan

def recursive_task_execution(prompt, messages, single_agent_plan, model_name, decomp_freq_penalty, decomposed_plan_code):
    print (f"\n\n*******************************************************************************************************************************")
    print ("Testing Task Excection...")
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
            _, text = LM(messages, model_name, max_tokens, frequency_penalty=decomp_freq_penalty, quantize=args.quantize, quantization_bits=args.quantization_bits) #1600 for gpt4o

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
                _, text = LM(messages, args.hf_model, max_tokens, frequency_penalty=decomp_freq_penalty, quantize=args.quantize, quantization_bits=args.quantization_bits) #1600 for gpt4o

                print(f"\n############################################# Updated LLM Response for Decomposed Plan #############################################################")
                print(text)
                user_input = input("\nIs this plan correct now? (yes/no): ")

            # user confirms the plan is correct
            print("Code Confirmed. Proceeding...")
            decomposed_plan_code = text
            
            ######################### Single Agent Code Prep #################################
            single_agent_plan = single_agent_code_prep(decomposed_plan_code, args.hf_model)
            decomposed_plan_code = recursive_task_execution(
                prompt, messages, single_agent_plan, model_name, 
                decomp_freq_penalty, decomposed_plan_code
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

    print ("Model Config Set.")

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
    # print (f"\n############################################# Provided Prompt #############################################################")
    # print (prompt)

    messages = [{"role": "system", "content": prompt.split('\n\n')[0]},
                {"role": "user", "content": prompt.split('\n\n')[1]},
                {"role": "assistant", "content": prompt.split('\n\n')[2]},
                {"role": "user", "content": prompt.split('\n\n')[3]}]
    _, text = LM(messages, args.hf_model, max_tokens, frequency_penalty=decomp_freq_penalty, quantize=args.quantize, quantization_bits=args.quantization_bits)

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
    prompt += '''
    IMPORTANT: Some objects are created from a source object after certain actions.
    For example, a Potato becomes PotatoSliced after the Slice action.
    Object Types marked with (*) only appear after an interaction—for instance, an Apple becomes AppleSliced after slicing.
    Objects with a (*) in their properties dictionary are not present in the scene initially but can be created after performing actions on existing objects.
    However, toasting does not create a new object type.
    For example, Bread remains Bread after being toasted; it does not become BreadToasted*.
    Additionally, note that both PutObject and DropObject functions already include PickUpObject and GoToObject.
    Therefore, you do not need to call PickUpObject or GoToObject separately before using PutObject or DropObject.
    '''

    print (f"\n\n*******************************************************************************************************************************")
    print ("Generating Decomposed Plans...")
    # print (f"\n############################################# Provided Prompt #############################################################")
    # print(prompt)

    messages = [{"role": "system", "content": prompt.split('\n\n')[0]},
                {"role": "user", "content": prompt.split('\n\n')[1]},
                {"role": "assistant", "content": prompt.split('\n\n')[2]},
                {"role": "user", "content": prompt.split('\n\n')[3]}]
    _, text = LM(messages, args.hf_model, max_tokens, frequency_penalty=decomp_freq_penalty, quantize=args.quantize, quantization_bits=args.quantization_bits) #1600 for gpt4o

    # decomposed_plan = text
    print (f"\n############################################# LLM Response #############################################################")
    print(text)
    
    # Ask the user for feedback
    user_input = input("\nIs this plan correct? (yes/no): ")
    while user_input.lower() != "yes":
        additional_instructions = input("Please provide more instructions to correct the plan: ")
        prompt += f"\nAdditional user instructions: {additional_instructions}"

        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": additional_instructions})
        _, text = LM(messages, args.hf_model, max_tokens, frequency_penalty=decomp_freq_penalty, quantize=args.quantize, quantization_bits=args.quantization_bits) #1600 for gpt4o

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
    prompt += '''
    IMPORTANT: Some objects are created from a source object after certain actions.
    For example, a Potato becomes PotatoSliced after the Slice action.
    Object Types marked with (*) only appear after an interaction—for instance, an Apple becomes AppleSliced after slicing.
    Objects with a (*) in their properties dictionary are not present in the scene initially but can be created after performing actions on existing objects.
    However, toasting does not create a new object type.
    For example, Bread remains Bread after being toasted; it does not become BreadToasted*.
    Additionally, note that both PutObject and DropObject functions already include PickUpObject and GoToObject.
    Therefore, you do not need to call PickUpObject or GoToObject separately before using PutObject or DropObject.
    '''
    prompt+= f"\nIMPORTANT: Always use the CleanObject function when performing a cleaning action. The parameter <toolObjectId> must be provided and should correspond to a cleaning tool available in the scene. The parameter <canBeUsedUpDetergentId> is optional—include it only if there is a Detergent in the scene with 'canBeUsedUp'=True and 'UsedUp'=False. If no such Detergent exists, then <toolObjectId> alone is sufficient."

    print (f"\n\n*******************************************************************************************************************************")
    print ("Generating Decomposed Plans CODE...")
    # print (f"\n############################################# Provided Prompt #############################################################")
    # print(prompt)

    messages = []
    for i,split_prompt in enumerate(prompt.split('***')):
        if i == 0:
            messages.append({"role": "system", "content": split_prompt})
        elif i%2 == 0:
            messages.append({"role": "assistant", "content": split_prompt})
        else:
            messages.append({"role": "user", "content": split_prompt})
    _, text = LM(messages, args.hf_model, max_tokens, frequency_penalty=decomp_code_freq_penalty, quantize=args.quantize, quantization_bits=args.quantization_bits)   

    # decomposed_plan_code = text
    print (f"\n############################################# LLM Response for Decomposed Plan #############################################################")
    print(text)

    # Ask the user for feedback
    user_input = input("\nIs this code plan correct? (yes/no): ")
    while user_input.lower() != "yes":
        additional_instructions = input("Please provide more instructions to correct the plan: ")
        prompt += f"\nAdditional user instructions: {additional_instructions}"

        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": additional_instructions})
        _, text = LM(messages, args.hf_model, max_tokens, frequency_penalty=decomp_freq_penalty, quantize=args.quantize, quantization_bits=args.quantization_bits) #1600 for gpt4o

        print(f"\n############################################# Updated LLM Response for Decomposed Plan #############################################################")
        print(text)
        user_input = input("\nIs this plan correct now? (yes/no): ")

    # user confirms the plan is correct
    print("Code Confirmed. Proceeding...")
    decomposed_plan_code = text

    ############################################## Single Agent Code Prep #################################################################
    single_agent_plan = single_agent_code_prep(decomposed_plan_code, args.hf_model)

    ################################################# Task Execution Test #################################################################
    decomposed_plan_code = recursive_task_execution(prompt, messages, single_agent_plan, args.hf_model, decomp_freq_penalty,
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
    messages = [{"role": "system", "content": allocate_prompt},
                {"role": "user", "content": prompt}]
    _, text = LM(messages, args.hf_model, max_tokens, frequency_penalty=0.02, quantize=args.quantize, quantization_bits=args.quantization_bits) #0.30 #1800 for gpt4o

    print (f"\n############################################# LLM Response #############################################################")
    print(text)

    # Ask the user for feedback
    user_input = input("\nIs this allocated plan code correct? (yes/no): ")
    while user_input.lower() != "yes":
        additional_instructions = input("Please provide more instructions to correct the plan: ")
        prompt += f"\nAdditional user instructions: {additional_instructions}"

        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": additional_instructions})
        _, text = LM(messages, args.hf_model, max_tokens, frequency_penalty=0.02, quantize=args.quantize, quantization_bits=args.quantization_bits)

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
