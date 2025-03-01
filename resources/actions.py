ai2thor_actions = ["GoToObject <robot><object>", "OpenObject <robot><object>", "CloseObject <robot><object>", 
                   "BreakObject <robot><object>", "SliceObject <robot><object>", "SwitchOn <robot><object>", 
                   "SwitchOff <robot><object>", "CleanObject <robot><object>", "PickupObject <robot><object>", 
                   "PutObject <robot><object><receptacleObject>", "DropHandObject <robot><object>", 
                   "ThrowObject <robot><object>", "PushObject <robot><object>", "PullObject <robot><object>"]
ai2thor_actions = ', '.join(ai2thor_actions)

actions = [
        "OpenObject <objectId>", 
        "CloseObject <objectId>", 
        "BreakObject <objectId>", 
        "SliceObject <objectId><toolObjectId>", 
        "SwitchOn <objectId>", 
        "SwitchOff <objectId>", 
        "CleanObject <objectId><toolObjectId><canBeUsedUpDetergentId>", 
        "PutObject <objectId><receptacleObjectId>", 
        "ThrowObject <objectId>"
        ]
actions = ', '.join(actions)
