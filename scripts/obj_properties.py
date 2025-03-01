import requests
from bs4 import BeautifulSoup
import pandas as pd

url = 'https://ai2thor.allenai.org/ithor/documentation/objects/object-types/'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

table = soup.findAll('table')
df = pd.read_html(str(table))
df = df[1]

context_interactions = {}
for i in range(2,len(df[1])):
    if type(df.iloc[i][5]) == str:
        context_interactions[df.iloc[i][0]] = df.iloc[i][5]
for key, value in context_interactions.items():
    print(f"{key}: {value}")


actionable_properties = {}
actionable_properties['AlarmClock'] = [df.iloc[2][2].lower()]
for i in range(2,len(df[1])):
    if type(df.iloc[i][2]) == str:
        actionable_properties[df.iloc[i][0]] = df.iloc[i][2].lower().split(',')
    else:
        actionable_properties[df.iloc[i][0]] = df.iloc[i][2]


for key, value in actionable_properties.items():
    if type(value) == list:
        for i,v in enumerate(value):
            v_cleaned = v.replace('(some)', '').strip()
            value[i] = v_cleaned 
            if v_cleaned == 'fillable':
                value[i] = 'canFillWithLiquid'
            elif v_cleaned == 'usedup':
                value[i] = 'canBeUsedUp'

for key, value in actionable_properties.items():
    print(f"{key}: {value}")

property_mapping = {
    'toggleable': ['isToggled'],
    'receptacle': ['receptacleObjectIds'],
    'moveable': ['isMoving'],
    'openable': ['isOpen', 'openness'],
    'breakable': ['isBroken'],
    'canFillWithLiquid': ['isFilledWithLiquid', 'fillLiquid'],
    'dirtyable': ['isDirty'],
    'canBeUsedUp': ['isUsedUp'],
    'cookable': ['isCooked', 'temperature'],
    'sliceable': ['isSliced'],
    'pickupable': ['isPickedUp']
}

for key, properties in actionable_properties.items():
    if type(properties) == list:
        extended_properties = set(properties)  # Use a set to avoid duplicates
        for prop in properties:
            if prop in property_mapping:
                extended_properties.update(property_mapping[prop])
        actionable_properties[key] = list(extended_properties)

for key, value in actionable_properties.items():
    print(f"{key}: {value}")

print(actionable_properties)
print('******************************************************************')
print(context_interactions)