import json
with open('../data/captions_train2014.json', 'rt') as f:
  j1 = json.load(f)

with open('../data/captions_val2014.json', 'rt') as f:
  j2 = json.load(f)

assert(j1['info'] == j2['info'])
assert(j1['type'] == j2['type'])
assert(j1['licenses'] == j2['licenses'])

j1['images'] = j1['images'] + j2['images']
j1['annotations'] = j1['annotations'] + j2['annotations']

with open('../data/captions_trainval2014.json', 'wt') as f:
  json.dump(j1, f)
