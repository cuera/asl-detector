import json
import pickle

m = pickle.load(open('model.p', 'rb'))['model']
letters = [chr(65 + int(c)) for c in m.classes_]
trees = []
for e in m.estimators_:
    t = e.tree_
    trees.append({'l': t.children_left.tolist(), 'r': t.children_right.tolist(),
                  'f': t.feature.tolist(), 't': [round(float(x), 5) for x in t.threshold],
                  'c': t.value[:, 0, :].argmax(1).tolist()})
json.dump({'letters': letters, 'trees': trees}, open('web/model.json', 'w'), separators=(',', ':'))
