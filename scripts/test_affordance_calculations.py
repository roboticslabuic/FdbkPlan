import ast

# agents and affordances
class Affordance:
    def __init__(self, skills):
        self.skills = skills

    def get_affordance(self, skill_name):
        skill = next((s for s in self.skills if skill_name in s), None)
        if skill:
            return skill[skill_name]['s_rate'] * skill[skill_name]['accuracy']
        return 0

    def open_object(self, obj):
        return self.get_affordance('GoToObject') * self.get_affordance('OpenObject')
    
    def close_object(self, obj):
        return self.get_affordance('GoToObject') * self.get_affordance('CloseObject')
    
    def break_object(self, obj):
        return self.get_affordance('GoToObject') * self.get_affordance('BreakObject')
    
    def clean_object(self, obj):
        return self.get_affordance('GoToObject') * self.get_affordance('CleanObject')
    
    def drophand_object(self, obj):
        return self.get_affordance('GoToObject') * self.get_affordance('DropHandObject')
    
    def throw_object(self, obj):
        return self.get_affordance('GoToObject') * self.get_affordance('ThrowObject')
    
    def switch_on(self, obj):
        return self.get_affordance('GoToObject') * self.get_affordance('SwitchOn')

    def switch_off(self, obj):
        return self.get_affordance('GoToObject') * self.get_affordance('SwitchOff')

    def put_object(self, obj1, obj2):
        return (self.get_affordance('GoToObject') * self.get_affordance('PickupObject') *
                self.get_affordance('GoToObject') * self.get_affordance('PutObject'))

    def slice_object(self, obj):
        return (self.get_affordance('GoToObject') * self.get_affordance('PickupObject') *
                self.get_affordance('GoToObject') * self.get_affordance('SliceObject'))

def extract_actions_from_code(code):
    tree = ast.parse(code)
    actions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            action = {}
            if isinstance(node.func, ast.Name):
                action_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                action_name = node.func.attr
            action['action'] = action_name
            if action_name == 'PutObject':
                action['object1'] = node.args[0].s
                action['object2'] = node.args[1].s
            elif action_name == 'SliceObject':
                action['object'] = node.args[0].s
            elif action_name in ['OpenObject', 'CloseObject', 'SwitchOn', 'SwitchOff', 'BreakObject', 'CleanObject', 'DropHandObject', 'ThrowObject']:
                action['object'] = node.args[0].s
            actions.append(action)
    return actions

def calculate_action_affordance(agent_affordance, action):
    if action['action'] == 'OpenObject':
        return agent_affordance.open_object(action['object'])
    elif action['action'] == 'BreakObject':
        return agent_affordance.break_object(action['object'])
    elif action['action'] == 'CleanObject':
        return agent_affordance.clean_object(action['object'])
    elif action['action'] == 'DropHandObject':
        return agent_affordance.drophand_object(action['object'])
    elif action['action'] == 'ThrowObject':
        return agent_affordance.throw_object(action['object'])
    elif action['action'] == 'CloseObject':
        return agent_affordance.close_object(action['object'])
    elif action['action'] == 'SwitchOn':
        return agent_affordance.switch_on(action['object'])
    elif action['action'] == 'SwitchOff':
        return agent_affordance.switch_off(action['object'])
    elif action['action'] == 'SliceObject':
        return agent_affordance.slice_object(action['object'])
    elif action['action'] == 'PutObject':
        return agent_affordance.put_object(action['object1'], action['object2'])

# Human and Robot Skills Dictionaries
human_skills = [
    {'GoToObject': {'s_rate': 1.0, 'accuracy': 0.99}},
    {'OpenObject': {'s_rate': 0.6, 'accuracy': 0.6}},
    {'CloseObject': {'s_rate': 0.6, 'accuracy': 0.6}},
    {'BreakObject': {'s_rate': 0.8, 'accuracy': 0.8}},
    {'SliceObject': {'s_rate': 0.8, 'accuracy': 0.7}},
    {'SwitchOn': {'s_rate': 0.9, 'accuracy': 0.9}},
    {'SwitchOff': {'s_rate': 0.9, 'accuracy': 0.9}},
    {'PickupObject': {'s_rate': 0.8, 'accuracy': 0.8}},
    {'PutObject': {'s_rate': 0.9, 'accuracy': 0.8}},
    {'DropHandObject': {'s_rate': 1.0, 'accuracy': 0.9}},
    {'ThrowObject': {'s_rate': 1.0, 'accuracy': 0.8}},
    {'PushObject': {'s_rate': 0.8, 'accuracy': 0.9}},
    {'PullObject': {'s_rate': 0.8, 'accuracy': 0.9}},
    {'CleanObject': {'s_rate': 0.8, 'accuracy': 0.8}}
]

robot_skills = [
    {'GoToObject': {'s_rate': 0.9, 'accuracy': 0.8}},
    {'OpenObject': {'s_rate': 0.8, 'accuracy': 0.8}},
    {'CloseObject': {'s_rate': 0.8, 'accuracy': 0.8}},
    {'PickupObject': {'s_rate': 0.8, 'accuracy': 0.8}},
    {'PutObject': {'s_rate': 0.8, 'accuracy': 0.8}},
    {'DropHandObject': {'s_rate': 0.9, 'accuracy': 0.8}},
    {'CleanObject': {'s_rate': 0.9, 'accuracy': 0.8}}
]

agents = [{'name': 'agent1', 'type': 'human'}, {'name': 'agent2', 'type': 'robot'}]

# Initialize Affordance Classes for Human and Robot Agents
human_agent_affordance = Affordance(human_skills)
robot_agent_affordance = Affordance(robot_skills)

# The slice_ingredients function code as a string
slice_ingredients_code = """
def slice_ingredients():
    # SubTask 1: Slice the Lettuce, Tomato, and Bread using a Knife. 
    # Given the state of the Lettuce, Tomato, and Bread, they are slicable but not sliced. They can be sliced using a Knife.
    # Given the state of the Knife, it is in the Drawer|-03.12|+00.78|+02.92 . To grab the knife, first the Drawer|-03.12|+00.78|+02.92 must be opened.
    # Given the state of the Knife, it is dirty. It must be washed before using.

    # 1: Open the Drawer.
    OpenObject('Drawer|-03.12|+00.78|+02.92')

    # 2: Get the Knife from the deawer and place it in the Sink.
    PutObject('Knife', 'Sink')

    # 3: Close the Drawer.
    CloseObject('Drawer|-03.12|+00.78|+02.92')

    # 4: Switch on the Faucet to clean the Knife.
    SwitchOn('Faucet')

    # 5: Wait for a while to let the Knife clean.
    time.sleep(5)

    # 6: Switch off the Faucet
    SwitchOff('Faucet')

    # 7: Get the Knife from the sink and Slice the Lettuce.
    SliceObject('Lettuce')

    # 8: Slice the Tomato.
    SliceObject('Tomato')

    # 9: Slice the Bread.
    SliceObject('Bread')

    # 10: Put the Knife back on the CounterTop.
    PutObject('Knife', 'CounterTop')
"""

# Extract actions from the code
actions = extract_actions_from_code(slice_ingredients_code)

# Calculate Affordances for Human and Robot Agents
human_affordances = [calculate_action_affordance(human_agent_affordance, action) for action in actions]
robot_affordances = [calculate_action_affordance(robot_agent_affordance, action) for action in actions]

# Print the affordances for each action
for i, action in enumerate(actions):
    action['human_affordances'] = human_affordances[i]
    action['robot_affordances'] = robot_affordances[i]
    # print(f"Action: {action}")
    # print(f"  Human Affordance Score: {human_affordances[i]}")
    # print(f"  Robot Affordance Score: {robot_affordances[i]}")
print (actions)
