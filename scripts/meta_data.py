import ai2thor.controller
from ai2thor.platform import CloudRendering
import argparse
import sys
sys.path.append(".")
from resources.assess_objectType import actionable_properties
from resources.assess_objectType import context_interactions

def print_data(floor_plan_id):
    controller = ai2thor.controller.Controller(scene="FloorPlan"+str(floor_plan_id), platform=CloudRendering)
    # print (controller.last_event.metadata.keys())
    # obj = list(set([obj["objectId"] for obj in controller.last_event.metadata["objects"]]))
    obj = list(set([obj["objectType"] for obj in controller.last_event.metadata["objects"]]))

    # print list of all types of objects in the scene.
    for each in obj:
        print(each)
    print('**************************************************************************')

    # print all objects in the scene alongside their metadata
    # print(controller.last_event.metadata['objects'])
    # print('**************************************************************************')

    # print specific objects in the scene alongside their metadata
    spec_objects = ['SoapBottle']
    for object in controller.last_event.metadata['objects']:
        for spec_object in spec_objects:
            if spec_object in object['objectType']:
                print('\n', object)
    print('**************************************************************************')

    # extract and print desired states of all objects in the scene.
    # objs_state = []
    # for each_obj in obj:
    #     for object in controller.last_event.metadata['objects']:
    #         if each_obj == object['objectType']:
    #             state_summary = {}
    #             state_summary['objectId'] = object['objectId']
    #             state_summary['parentReceptacles'] = []
    #             if object['parentReceptacles']:
    #                 state_summary['parentReceptacles'] = [object['parentReceptacles'][0]]
    #             if actionable_properties[each_obj]:
    #                 for each_key in actionable_properties[each_obj]:
    #                     state_summary[each_key] = object[each_key]
    #             objs_state.append(state_summary)

    # for each in objs_state:
    #     print('\n\n', each)
    # print('**************************************************************************')

    # extract and print desired contextual interactions of all objects in the scene.
    # objs_context = []
    # for each_obj in obj:
    #     if each_obj in context_interactions:
    #         print(each_obj)
    #         context_summary = {}
    #         context_summary['objectType'] = each_obj
    #         context_summary['contextual_interactions'] = context_interactions[each_obj]
    #         objs_context.append(context_summary)
    # for each in objs_context:
    #     print('\n\n', each)
    # print('**************************************************************************')

    # perform desired action and print state of the object of interest after performing the desired action
    # controller.step(action="SliceObject", objectId='Bread|-01.52|+00.96|+00.28', forceAction=True)
    # for object in controller.last_event.metadata['objects']:
    #     if 'SinkBasin' in object['objectType']:
    #         object['receptacleObjectIds'] = []

    # for object in controller.last_event.metadata['objects']:
    #     if 'Sink' in object['name']:
    #         print(object, '\n')
    # print('**************************************************************************')


    # extract and print actionable properties of only needed objects.
    # needed_objects =  ['Egg' , 'EggCracked', 'Tomato', 'TomatoSliced', 'Bread', 'BreadSliced', 'Knife', 'Pan', 'Plate', 'StoveBurner', 'Sink']
    # needed_objects_props = {}
    # for obj in needed_objects:
    #     for item in actionable_properties:
    #         if obj in item:
    #             needed_objects_props[item] = actionable_properties[item]
    # print(needed_objects_props)
    # print('**************************************************************************')

    # extract and print desired states of only needed objects.
    # objs_state = []
    # for each_obj in needed_objects:
    #     for object in controller.last_event.metadata['objects']:
    #         if each_obj in object['objectType']:
    #             state_summary = {}
    #             state_summary['objectId'] = object['objectId']
    #             state_summary['parentReceptacles'] = []
    #             if object['parentReceptacles']:
    #                 state_summary['parentReceptacles'] = [object['parentReceptacles'][0]]
    #             if actionable_properties[each_obj]:
    #                 for each_key in actionable_properties[each_obj]:
    #                     state_summary[each_key] = object[each_key]
    #             objs_state.append(state_summary)

    # for each in objs_state:
    #     print('\n\n', each)
    # print('**************************************************************************')  
    
    # extract and print desired contextual interactions of only needed objects.
    # objs_context = {}
    # for each_obj in needed_objects:
    #     for key, value in context_interactions.items():
    #         if each_obj in key:
    #             print(each_obj)
    #             objs_context[key] = value
    # print(objs_context)
    # print('**************************************************************************')

    # for each in controller.last_event.metadata["objects"]:
    #     print(each,'\n\n')

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor-plan", type=int, required=True)
    args = parser.parse_args()
    print_data(args.floor_plan)