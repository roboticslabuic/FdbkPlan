
for i in range(2):
    action_queue({'action':'Done'})
    time.sleep(0.1)


time.sleep(5)


objs = list([obj for obj in c.last_event.metadata["objects"]])

with open("final_states.txt", "w") as file:
    success_rate = success_exec/total_exec
    total_exec = f"Total number of action executions = {total_exec}\n\n"
    file.write(total_exec)
    success_exec = f"Total number of successful executions = {success_exec}\n\n"
    file.write(success_exec)
    success_rate = f"Success Rate = {success_rate}\n\n"
    file.write(success_rate)
    print(total_exec, success_exec, success_rate)

    header = "\n**************************************** Final States ***********************************************\n"
    header += "\n\nFinal State of the World After Executing the Plan:\n"
    print(header)
    file.write(header)
    
    # Loop through needed objects and find their final states
    for obj in needed_objects:
        for object in c.last_event.metadata['objects']:
            if obj in object['name']:
                # Format the object details for printing and saving
                object_details = f"{object}\n\n"
                print(object_details)
                file.write(object_details)


# print(f"\n**************************************** Final States ***********************************************\n")
# print ("\n\nFinal State of the World After Executing the Plan:")
# for obj in needed_objects:
#     for object in c.last_event.metadata['objects']:
#         if obj in object['name']:
#             print(object,'\n\n')
# for object in c.last_event.metadata['objects']:
#     if 'Sink' in object['name']:
#         print(object,'\n\n')
print(f"\n**************************************** Generating Videos ***********************************************\n")
generate_video()