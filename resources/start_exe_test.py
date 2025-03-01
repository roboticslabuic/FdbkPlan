import math
import re
import shutil
import subprocess
import time
import threading
import numpy as np
import cv2
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from scipy.spatial import distance
from typing import Tuple
from collections import deque
import random
import os
from glob import glob
import argparse
import sys

def closest_node(node, nodes, num_agents, clost_node_location):
    crps = []
    distances = distance.cdist([node], nodes)[0]
    dist_indices = np.argsort(np.array(distances))
    for i in range(num_agents):
        pos_index = dist_indices[(i * 5) + clost_node_location[i]]
        crps.append (nodes[pos_index])
    return crps

def distance_pts(p1: Tuple[float, float, float], p2: Tuple[float, float, float]):
    return ((p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5

def generate_video():
    frame_rate = 5
    # input_path, prefix, char_id=0, image_synthesis=['normal'], frame_rate=5
    # cur_path = './videos/' + os.path.dirname(__file__) + "/*/"
    cur_path = './videos/' + "*/"
    for imgs_folder in glob(cur_path, recursive = False):
        view = imgs_folder.split('/')[-2]
        if not os.path.isdir(imgs_folder):
            print("The input path: {} you specified does not exist.".format(imgs_folder))
        else:
            command_set = ['ffmpeg', '-i',
                                '{}/img_%05d.png'.format(imgs_folder),
                                '-framerate', str(frame_rate),
                                '-r', str(frame_rate),
                                '-pix_fmt', 'yuv420p',
                                '-crf', '18',  # Lower CRF for better quality
                                '{}/video_{}.mp4'.format('./videos/', view)]
                                # '{}/video_{}.mp4'.format('./videos/' + os.path.dirname(__file__), view)]
            subprocess.call(command_set)


agents = [{'name': 'agent1', 'type': 'robot'}]

parser = argparse.ArgumentParser()
parser.add_argument("--floor-plan", type=int, required=True)
args = parser.parse_args()

floor_no = args.floor_plan

c = Controller( height=1000, width=1000, platform=CloudRendering)
c.reset("FloorPlan" + str(floor_no)) 
num_agents = len(agents)

# initialize n agents into the scene
multi_agent_event = c.step(dict(action='Initialize', agentMode="default", snapGrid=False, gridSize=0.5, rotateStepDegrees=20, visibilityDistance=100, fieldOfView=90, agentCount=num_agents))

# add a top view camera
event = c.step(action="GetMapViewCameraProperties")
event = c.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])

# get reachabel positions
reachable_positions_ = c.step(action="GetReachablePositions").metadata["actionReturn"]
reachable_positions = positions_tuple = [(p["x"], p["y"], p["z"]) for p in reachable_positions_]

# randomize postions of the agents
for i in range (num_agents):
    init_pos = random.choice(reachable_positions_)
    c.step(dict(action="Teleport", position=init_pos, agentId=i))
    
objs = list([obj["objectId"] for obj in c.last_event.metadata["objects"]])

for i in range (num_agents):
    multi_agent_event = c.step(action="LookDown", degrees=35, agentId=i)
    # c.step(action="LookUp", degrees=30, 'agent_id':i)

# delete if current output already exist
cur_path = './videos' + "/*/"
for x in glob(cur_path, recursive = True):
    shutil.rmtree (x)

# create new folders to save the images from the agents
for i in range(num_agents):
    folder_name = "agent_" + str(i+1)
    folder_path = './videos/' + folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# create folder to store the top view images
folder_name = "top_view"
folder_path = './videos/' + folder_name
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

global total_exec, success_exec, task_over, img_counter
img_counter = 0
total_exec = 0
success_exec = 0
recp_id = None

def action_queue(act):
    global total_exec, success_exec, task_over, img_counter    
    try:
        if act['action'] == 'ObjectNavExpertAction':
            multi_agent_event = c.step(dict(action=act['action'], position=act['position'], agentId=act['agent_id']))
            next_action = multi_agent_event.metadata['actionReturn']

            if next_action != None:
                multi_agent_event = c.step(action=next_action, agentId=act['agent_id'], forceAction=True)
        
        elif act['action'] == 'MoveAhead':
            multi_agent_event = c.step(action="MoveAhead", agentId=act['agent_id'])
            
        elif act['action'] == 'MoveBack':
            multi_agent_event = c.step(action="MoveBack", agentId=act['agent_id'])
                
        elif act['action'] == 'RotateLeft':
            multi_agent_event = c.step(action="RotateLeft", degrees=act['degrees'], agentId=act['agent_id'])
            
        elif act['action'] == 'RotateRight':
            multi_agent_event = c.step(action="RotateRight", degrees=act['degrees'], agentId=act['agent_id'])
            
        elif act['action'] == 'PickupObject':
            total_exec += 1
            multi_agent_event = c.step(action="PickupObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
            if multi_agent_event.metadata['errorMessage'] != "":
                raise RuntimeError(f"PickupObject error: {multi_agent_event.metadata['errorMessage']}")
            else:
                success_exec += 1

        elif act['action'] == 'PutObject':
            total_exec += 1
            multi_agent_event = c.step(action="PutObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
            if multi_agent_event.metadata['errorMessage'] != "":
                raise RuntimeError(f"PutObject error: {multi_agent_event.metadata['errorMessage']}")
            else:
                success_exec += 1

        elif act['action'] == 'ToggleObjectOn':
            total_exec += 1
            multi_agent_event = c.step(action="ToggleObjectOn", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
            if multi_agent_event.metadata['errorMessage'] != "":
                raise RuntimeError(f"ToggleObjectOn error: {multi_agent_event.metadata['errorMessage']}")
            else:
                success_exec += 1
        
        elif act['action'] == 'ToggleObjectOff':
            total_exec += 1
            multi_agent_event = c.step(action="ToggleObjectOff", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
            if multi_agent_event.metadata['errorMessage'] != "":
                raise RuntimeError(f"ToggleObjectOff error: {multi_agent_event.metadata['errorMessage']}")
            else:
                success_exec += 1
            
        elif act['action'] == 'OpenObject':
            total_exec += 1
            multi_agent_event = c.step(action="OpenObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
            if multi_agent_event.metadata['errorMessage'] != "":
                raise RuntimeError(f"OpenObject error: {multi_agent_event.metadata['errorMessage']}")
            else:
                success_exec += 1
   
        elif act['action'] == 'CloseObject':
            total_exec += 1
            multi_agent_event = c.step(action="CloseObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
            if multi_agent_event.metadata['errorMessage'] != "":
                raise RuntimeError(f"CloseObject error: {multi_agent_event.metadata['errorMessage']}")
            else:
                success_exec += 1

        elif act['action'] == 'UseUpObject':
            total_exec += 1
            print("$$$$$$$$$$$$ Check UseUpObject $$$$$$$$$$$$$$$$$")
            print(act['objectId'])
            multi_agent_event = c.step(action="UseUpObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
            if multi_agent_event.metadata['errorMessage'] != "":
                raise RuntimeError(f"UseUpObject error: {multi_agent_event.metadata['errorMessage']}")
            else:
                success_exec += 1
                
        elif act['action'] == 'SliceObject':
            total_exec += 1
            multi_agent_event = c.step(action="SliceObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
            if multi_agent_event.metadata['errorMessage'] != "":
                raise RuntimeError(f"SliceObject error: {multi_agent_event.metadata['errorMessage']}")
            else:
                success_exec += 1
                
        elif act['action'] == 'ThrowObject':
            total_exec += 1
            multi_agent_event = c.step(action="ThrowObject", moveMagnitude=100, agentId=act['agent_id'], forceAction=True)
            if multi_agent_event.metadata['errorMessage'] != "":
                raise RuntimeError(f"ThrowObject error: {multi_agent_event.metadata['errorMessage']}")
            else:
                success_exec += 1

        elif act['action'] == 'DropHandObject':
            total_exec += 1
            multi_agent_event = c.step(action="DropHandObject", agentId=act['agent_id'], forceAction=True)
            if multi_agent_event.metadata['errorMessage'] != "":
                raise RuntimeError(f"DropHandObject error: {multi_agent_event.metadata['errorMessage']}")
            else:
                success_exec += 1
                
        elif act['action'] == 'BreakObject':
            total_exec += 1
            multi_agent_event = c.step(action="BreakObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
            if multi_agent_event.metadata['errorMessage'] != "":
                raise RuntimeError(f"BreakObject error: {multi_agent_event.metadata['errorMessage']}")
            else:
                success_exec += 1

        elif act['action'] == 'CleanObject':
            total_exec += 1
            multi_agent_event = c.step(action="CleanObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
            if multi_agent_event.metadata['errorMessage'] != "" and not ('does not have dirtyable property' in multi_agent_event.metadata['errorMessage'] or 'is already Clean' in multi_agent_event.metadata['errorMessage']):
                raise RuntimeError(f"CleanObject error: {multi_agent_event.metadata['errorMessage']}")
            else:
                success_exec += 1

        elif act['action'] == 'Done':
            multi_agent_event = c.step(action="Done")

        img_counter += 1
        for i,eve in enumerate(multi_agent_event.events):
            f_name = './videos' + "/agent_" + str(i+1) + "/img_" + str(img_counter).zfill(5) + ".png"
            cv2.imwrite(f_name, eve.cv2img)
        top_view_rgb = cv2.cvtColor(c.last_event.events[0].third_party_camera_frames[-1], cv2.COLOR_BGR2RGB)
        f_name = './videos' + "/top_view/img_" + str(img_counter).zfill(5) + ".png"
        cv2.imwrite(f_name, top_view_rgb)

        return None
            
            
    except Exception as e:
        return e


def GoToObject(agents, dest_obj):
    global recp_id
    
    if not isinstance(agents, list):
        agents = [agents]
    num_agents = len (agents)

    # agents distance to the goal 
    dist_goals = [10.0] * len(agents)
    prev_dist_goals = [10.0] * len(agents)
    count_since_update = [0] * len(agents)
    clost_node_location = [0] * len(agents)
    
    # list of objects in the scene and their centers
    objs = list([obj["objectId"] for obj in c.last_event.metadata["objects"]])
    objs_type = list([obj["objectType"] for obj in c.last_event.metadata["objects"]])
    objs_center = list([obj["axisAlignedBoundingBox"]["center"] for obj in c.last_event.metadata["objects"]])
    if "|" in dest_obj:
        # objectId is given
        dest_obj_id = dest_obj
        # extract object position
        pos_arr = dest_obj_id.split("|")
        dest_obj_center = {'x': float(pos_arr[1]), 'y': float(pos_arr[2]), 'z': float(pos_arr[3])}
    else:
        # objectType is given
        for idx, obj_type in enumerate(objs_type):
            match = re.search(dest_obj, obj_type)
            if match is not None:
                dest_obj_id = objs[idx]
                dest_obj_center = objs_center[idx]
                if dest_obj_center != {'x': 0.0, 'y': 0.0, 'z': 0.0}:
                    break # find the first instance      
        
    dest_obj_pos = [dest_obj_center['x'], dest_obj_center['y'], dest_obj_center['z']]
    
    # closest reachable position for each agent
    # all agents cannot reach the same spot 
    # differt close points needs to be found for each agent
    crp = closest_node(dest_obj_pos, reachable_positions, num_agents, clost_node_location)
    
    goal_thresh = 0.25
    # at least one agent is far away from the goal
       
    printed_action = False
    while all(d > goal_thresh for d in dist_goals):
        for ia, agent in enumerate(agents):
            agent_name = agent['name']
            agent_type = agent['type']
            agent_id = int(agent_name[-1]) - 1
            if not printed_action:
                print ("Performing Action Goto ", dest_obj_id, ", ", agent_name, agent_type)
                printed_action = True
            print("|", end="", flush=True)

            # get the pose of agent        
            metadata = c.last_event.events[agent_id].metadata
            location = {
                "x": metadata["agent"]["position"]["x"],
                "y": metadata["agent"]["position"]["y"],
                "z": metadata["agent"]["position"]["z"],
                "rotation": metadata["agent"]["rotation"]["y"],
                "horizon": metadata["agent"]["cameraHorizon"]}
            
            prev_dist_goals[ia] = dist_goals[ia] # store the previous distance to goal
            dist_goals[ia] = distance_pts([location['x'], location['y'], location['z']], crp[ia])
            
            dist_del = abs(dist_goals[ia] - prev_dist_goals[ia])
            # print (ia, "Dist to Goal: ", dist_goals[ia], dist_del, clost_node_location[ia])
            if dist_del < 0.2:
                # agent did not move 
                count_since_update[ia] += 1
            else:
                # agent moving 
                count_since_update[ia] = 0
                
            if count_since_update[ia] < 8:
                action_queue({'action':'ObjectNavExpertAction', 'position':dict(x=crp[ia][0], y=crp[ia][1], z=crp[ia][2]), 'agent_id':agent_id})
            else:    
                #updating goal
                clost_node_location[ia] += 1
                count_since_update[ia] = 0
                crp = closest_node(dest_obj_pos, reachable_positions, num_agents, clost_node_location)
            if dist_goals[ia] <= goal_thresh:
                print (f"\nReached: ", dest_obj_id, ", ", agent_name, agent_type)   
            time.sleep(0.5)

    # align the agent once goal is reached
    # compute angle between agent heading and object
    metadata = c.last_event.events[agent_id].metadata
    agent_location = {
        "x": metadata["agent"]["position"]["x"],
        "y": metadata["agent"]["position"]["y"],
        "z": metadata["agent"]["position"]["z"],
        "rotation": metadata["agent"]["rotation"]["y"],
        "horizon": metadata["agent"]["cameraHorizon"]}
    
    agent_object_vec = [dest_obj_pos[0] -agent_location['x'], dest_obj_pos[2]-agent_location['z']]
    y_axis = [0, 1]
    unit_y = y_axis / np.linalg.norm(y_axis)
    unit_vector = agent_object_vec / np.linalg.norm(agent_object_vec)
    
    angle = math.atan2(np.linalg.det([unit_vector,unit_y]),np.dot(unit_vector,unit_y))
    angle = 360*angle/(2*np.pi)
    angle = (angle + 360) % 360
    rot_angle = angle - agent_location['rotation']
    
    if rot_angle > 0:
        action_queue({'action':'RotateRight', 'degrees':abs(rot_angle), 'agent_id':agent_id})
    else:
        action_queue({'action':'RotateLeft', 'degrees':abs(rot_angle), 'agent_id':agent_id})
        
    # print ("Finished Action Go to, Reached: ", dest_obj)
    # if ("Cabinet" in dest_obj) or ("Fridge" in dest_obj) or ("CounterTop" in dest_obj):
    #     recp_id = dest_obj_id
    
def PickupObject(agents, pick_obj):

    if not isinstance(agents, list):
        agents = [agents]
    num_agents = len (agents)
    # agents distance to the goal 
    for idx in range(num_agents):
        agent = agents[idx]
        agent_name = agent['name']
        agent_type = agent['type'] 
        agent_id = int(agent_name[-1]) - 1
        # list of objects in the scene and their centers
        objs = list([obj["objectId"] for obj in c.last_event.metadata["objects"]])
        objs_type = list([obj["objectType"] for obj in c.last_event.metadata["objects"]])
        objs_center = list([obj["axisAlignedBoundingBox"]["center"] for obj in c.last_event.metadata["objects"]])
        
        if '|' in pick_obj:
            pick_obj_id = pick_obj
        else:
            for idx, obj_type in enumerate(objs_type):
                match = re.search(pick_obj, obj_type)
                if match is not None:
                    pick_obj_id = objs[idx]
                    pick_obj_center = objs_center[idx]
                    if 'Sliced' in pick_obj:
                        pick_obj_id = objs[idx+1]
                        pick_obj_center = objs_center[idx+1]
                    if pick_obj_center != {'x': 0.0, 'y': 0.0, 'z': 0.0}:
                        break # find the first instance
        
        GoToObject(agent, pick_obj_id)
        print ("Performing Action PickUp ", pick_obj_id, ", ", agent_name, agent_type)
        e = action_queue({'action':'PickupObject', 'objectId':pick_obj_id, 'agent_id':agent_id})
        if e:
            print(f"{e}. Failed to PickUp {pick_obj_id}", file=sys.stderr)
            sys.exit(1)
        time.sleep(1)
    
def PutObject(agent, put_obj, recp):
    print(f"\n**************************************** Executing SKILL PUT ***********************************************\n")
    # global recp_id 
    agent_name = agent['name']
    agent_type = agent['type'] 
    agent_id = int(agent_name[-1]) - 1
    objs = list([obj["objectId"] for obj in c.last_event.metadata["objects"]])
    objs_type = list([obj["objectType"] for obj in c.last_event.metadata["objects"]])
    objs_center = list([obj["axisAlignedBoundingBox"]["center"] for obj in c.last_event.metadata["objects"]])
    # objs_dists = list([obj["distance"] for obj in c.last_event.metadata["objects"]])
    objs_receps = list([obj["receptacleObjectIds"] for obj in c.last_event.metadata["objects"]])

    # metadata = c.last_event.events[agent_id].metadata
    # agent_location = [metadata["agent"]["position"]["x"], metadata["agent"]["position"]["y"], metadata["agent"]["position"]["z"]]
    # dist_to_recp = 9999999 # distance b/w agent and the recp obj
    if '|' in recp:
        recp_obj_id = recp
    else:
        for idx, obj_type in enumerate(objs_type):
            match = re.search(recp, obj_type)
            if match is not None:
                #dist = objs_dists[idx]
                #if dist < dist_to_recp:
                recp_obj_id = objs[idx]
                recp_obj_center = objs_center[idx]
                obj_recep = objs_receps[idx]
                if 'StoveBurner' in obj_type:
                    if recp_obj_center != {'x': 0.0, 'y': 0.0, 'z': 0.0} and not obj_recep:
                        break # find the first empty StoveBurner
                        #dist_to_recp = dist
                else:
                    if recp_obj_center != {'x': 0.0, 'y': 0.0, 'z': 0.0}:
                        break # find the first instance
                        #dist_to_recp = dist               
                           
    # if recp_id is not None:
    #     recp_obj_id = recp_id

    PickupObject(agent, put_obj)
    GoToObject(agent, recp_obj_id)
    # time.sleep(1)
    #print ("Putting Down ", put_obj, recp_obj_id, dest_obj_center)
    print ("Performing Action Put ", put_obj, recp_obj_id, ", ", agent_name, agent_type)
    e = action_queue({'action':'PutObject', 'heldObject': put_obj, 'objectId':recp_obj_id, 'agent_id':agent_id})
    if e:
        print(f"{e}. Failed to Put {put_obj} on {recp_obj_id}", file=sys.stderr)
        sys.exit(1)
    time.sleep(1)

         
def SwitchOn(agent, sw_obj):
    print(f"\n**************************************** Executing SKILL SWITCH_ON ***********************************************\n")
    # GoToObject(agent, sw_obj)
    agent_name = agent['name']
    agent_type = agent['type'] 
    agent_id = int(agent_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    # turn on all stove burner
    if "StoveKnob" in sw_obj:
        for obj in objs:
            match = re.search("StoveKnob", obj)
            if match is not None:
                sw_obj_id = obj
                GoToObject(agent, sw_obj_id)
                # time.sleep(1)
                print ("Performing Action SwitchOn ", sw_obj_id, ", ", agent_name, agent_type)
                e = action_queue({'action':'ToggleObjectOn', 'objectId':sw_obj_id, 'agent_id':agent_id})
                time.sleep(0.5)
                
    
    # all objects apart from Stove Burner
    else:
        if '|' in sw_obj:
            sw_obj_id = sw_obj
        else:
            for obj in objs:
                match = re.search(sw_obj, obj)
                if match is not None:
                    sw_obj_id = obj
                    break # find the first instance
        GoToObject(agent, sw_obj_id)
        # time.sleep(1)
        print ("Performing Action SwitchOn ", sw_obj_id, ", ", agent_name, agent_type) 
        e = action_queue({'action':'ToggleObjectOn', 'objectId':sw_obj_id, 'agent_id':agent_id})
        time.sleep(1)

    if e:
        print(f"{e}. Failed to SwitchOn {sw_obj_id}.", file=sys.stderr)
        sys.exit(1)
                   
        
def SwitchOff(agent, sw_obj):
    print(f"\n**************************************** Executing SKILL SWITCH_OFF ***********************************************\n")
    agent_name = agent['name']
    agent_type = agent['type']
    agent_id = int(agent_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    # turn on all stove burner
    if "StoveKnob" in sw_obj:
        for obj in objs:
            match = re.search("StoveKnob", obj)
            if match is not None:
                sw_obj_id = obj
                GoToObject(agent, sw_obj_id)
                print ("Performing Action SwitchOff ", sw_obj_id, ", ", agent_name, agent_type)
                e = action_queue({'action':'ToggleObjectOff', 'objectId':sw_obj_id, 'agent_id':agent_id})
                time.sleep(0.5)
    
    # all objects apart from Stove Burner
    else:
        if '|' in sw_obj:
            sw_obj_id = sw_obj
        else:
            for obj in objs:
                match = re.search(sw_obj, obj)
                if match is not None:
                    sw_obj_id = obj
                    break # find the first instance
        GoToObject(agent, sw_obj_id)
        print ("Performing Action SwitchOff ", sw_obj_id, ", ", agent_name, agent_type)
        e = action_queue({'action':'ToggleObjectOff', 'objectId':sw_obj_id, 'agent_id':agent_id})
        time.sleep(1)

    if e:
        print(f"{e}. Failed to SwitchOff {sw_obj_id}.", file=sys.stderr)
        sys.exit(1)

    
def OpenObject(agent, opn_obj):
    print(f"\n**************************************** Executing SKILL OPEN ***********************************************\n")
    agent_name = agent['name']
    agent_type = agent['type']
    agent_id = int(agent_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))

    if '|' in opn_obj:
        opn_obj_id = opn_obj
    else:
        for obj in objs:
            match = re.search(opn_obj, obj)
            if match is not None:
                opn_obj_id = obj
                break # find the first instance
        
    # global recp_id
    # if recp_id is not None:
    #     opn_obj_id = recp_id
    
    GoToObject(agent, opn_obj_id)
    print ("Performing Action Open ", opn_obj_id, ", ", agent_name, agent_type)
    e = action_queue({'action':'OpenObject', 'objectId':opn_obj_id, 'agent_id':agent_id})
    time.sleep(1)

    if e:
        print(f"{e}. Failed to Open {opn_obj_id}.", file=sys.stderr)
        sys.exit(1)

    
    
def CloseObject(agent, clz_obj):
    print(f"\n**************************************** Executing SKILL CLOSE ***********************************************\n")
    agent_name = agent['name']
    agent_type = agent['type']
    agent_id = int(agent_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))

    if '|' in clz_obj:
        clz_obj_id = clz_obj
    else:
        for obj in objs:
            match = re.search(clz_obj, obj)
            if match is not None:
                clz_obj_id = obj
                break # find the first instance
        
    # global recp_id
    # if recp_id is not None:
    #     clz_obj_id = recp_id
        
    GoToObject(agent, clz_obj_id)
    print ("Performing Action Close ", clz_obj_id, ", ", agent_name, agent_type)
    e = action_queue({'action':'CloseObject', 'objectId':clz_obj_id, 'agent_id':agent_id}) 
    # if recp_id is not None:
    #     recp_id = None
    time.sleep(1)

    if e:
        print(f"{e}. Failed to Close {clz_obj_id}.", file=sys.stderr)
        sys.exit(1)
    
def BreakObject(agent, brk_obj):
    print(f"\n**************************************** Executing SKILL BREAK ***********************************************\n")
    agent_name = agent['name']
    agent_type = agent['type']
    agent_id = int(agent_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))

    if '|' in brk_obj:
        brk_obj_id = brk_obj
    else:
        for obj in objs:
            match = re.search(brk_obj, obj)
            if match is not None:
                brk_obj_id = obj
                break # find the first instance
    GoToObject(agent, brk_obj_id)
    print ("Performing Action Break ", brk_obj_id, ", ", agent_name, agent_type)
    e = action_queue({'action':'BreakObject', 'objectId':brk_obj_id, 'agent_id':agent_id}) 
    time.sleep(1)

    if e:
        print(f"{e}. Failed to Break {brk_obj_id}.", file=sys.stderr)
        sys.exit(1)

    

def SliceObject(agent, slc_obj, tool_obj):
    print(f"\n**************************************** Executing SKILL SLICE ***********************************************\n")
    agent_name = agent['name']
    agent_type = agent['type']
    agent_id = int(agent_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))

    if '|' in slc_obj:
        slc_obj_id = slc_obj
    else:
        for obj in objs:
            match = re.search(slc_obj, obj)
            if match is not None:
                slc_obj_id = obj
                break # find the first instance

    PickupObject(agent, tool_obj)
    GoToObject(agent, slc_obj_id)
    print ("Performing Action Slice ", slc_obj_id, ", ", agent_name, agent_type)
    e = action_queue({'action':'SliceObject', 'heldTool': tool_obj, 'objectId':slc_obj_id, 'agent_id':agent_id})      
    time.sleep(1)
    if e:
        print(f"{e}. Failed to Slice {slc_obj_id}.", file=sys.stderr)
        sys.exit(1)
    print ("Performing Action Drop Knife, ", agent_name, agent_type)
    e = action_queue({'action':'DropHandObject', 'agent_id':agent_id}) 
    time.sleep(1)
    if e:
        print(f"{e}. Failed to Drop {tool_obj}.", file=sys.stderr)
        sys.exit(1)
    
def CleanObject(agent, cln_obj, tool_obj, clnsr_obj=None):
    print(f"\n**************************************** Executing SKILL CLEAN ***********************************************\n")
    agent_name = agent['name']
    agent_type = agent['type']
    agent_id = int(agent_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))

    if '|' in cln_obj:
        cln_obj_id = cln_obj
    else:
        for obj in objs:
            match = re.search(cln_obj, obj)
            if match is not None:
                cln_obj_id = obj
                break # find the first instance
    
    PickupObject(agent, tool_obj)
    if clnsr_obj:
        UseUpObject(agent, clnsr_obj)
    GoToObject(agent, cln_obj_id)
    print ("Performing Action Clean ", cln_obj_id, ", ", agent_name, agent_type)
    e = action_queue({'action':'CleanObject', 'objectId':cln_obj_id, 'agent_id':agent_id}) 
    time.sleep(1)
    if e:
        print(f"{e}. Failed to Clean {cln_obj_id}.", file=sys.stderr)
        sys.exit(1)
    print ("Performing Action Drop ", tool_obj, ", ", agent_name, agent_type)
    e = action_queue({'action':'DropHandObject', 'agent_id':agent_id})
    time.sleep(1)
    if e:
        print(f"{e}. Failed to Drop {tool_obj}.", file=sys.stderr)
        sys.exit(1)

def UseUpObject(agent, use_obj):
    agent_name = agent['name']
    agent_type = agent['type']
    agent_id = int(agent_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))

    if '|' in use_obj:
        use_obj_id = use_obj
    else:
        for obj in objs:
            match = re.search(use_obj, obj)
            if match is not None:
                use_obj_id = obj
                break # find the first instance
    
    GoToObject(agent, use_obj_id)
    print ("Performing Action UseUp ", use_obj_id, ", ", agent_name, agent_type)
    e = action_queue({'action':'UseUpObject', 'objectId':use_obj_id, 'agent_id':agent_id}) 
    time.sleep(1)
    if e:
        print(f"{e}. Failed to UseUp {use_obj_id}.", file=sys.stderr)
        sys.exit(1)

def ThrowObject(agent, thrw_obj):
    print(f"\n**************************************** Executing SKILL THROW ***********************************************\n")
    agent_name = agent['name']
    agent_type = agent['type']
    agent_id = int(agent_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    objs_type = list([obj["objectType"] for obj in c.last_event.metadata["objects"]])

    if '|' in thrw_obj:
        thrw_obj_id = thrw_obj
    else:
        for idx, obj_type in enumerate(objs_type):
            match = re.search(thrw_obj, obj_type)
            if match is not None:
                thrw_obj_id = objs[idx]
                if 'Sliced' in thrw_obj:
                    thrw_obj_id = objs[idx+1]
                break
    
    PickupObject(agent, thrw_obj_id)
    print ("Performing Action Throw ", thrw_obj_id, ", ", agent_name, agent_type)
    e = action_queue({'action':'ThrowObject', 'objectId': thrw_obj_id, 'agent_id':agent_id}) 
    time.sleep(1)
    if e:
        print(f"{e}. Failed to Throw {thrw_obj_id}.", file=sys.stderr)
        sys.exit(1)
