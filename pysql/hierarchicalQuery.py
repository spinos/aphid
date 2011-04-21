execfile('./sqlHierarchy.py')
print('qsl sample')
database_name = '/Users/jianzhang/Desktop/foo.db'
db = sqlHierarchy(database_name)
print 'add animals'
db.addLeaf('animal', 'carnivore')
db.addLeaf('animal', 'herbivore')
db.addLeaf('carnivore', 'lion')
db.addLeaf('carnivore', 'crocodile')
db.addLeaf('crocodile','alligator')
db.addLeaf('herbivore', 'horse')
db.addLeaf('carnivore', 'tiger')
db.addLeaf('carnivore', 'cheetah')
db.addLeaf('herbivore', 'cow')
db.addLeaf('animal', 'omnivore')
db.addLeaf('carnivore', 'wolf')
db.addLeaf('herbivore', 'zebra')
db.addLeaf('omnivore', 'human')
print '    display all animals'
db.displayTree('animal')
print '    display all meat eaters'
db.displayTree('carnivore')
print '    display all grazers'
db.displayTree('herbivore')
print '    display others'
db.displayTree('omnivore')
print '    display path to aligator'
db.displayPathTo('alligator')
