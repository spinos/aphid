import xml.dom.minidom as dom
import random

def create_leaf(doc, color):
    leaf = doc.createElement("leaf")
    text = doc.createTextNode(color)
    leaf.appendChild(text)
    return leaf

def create_branch(doc, direction):
    branch = doc.createElement("branch")
    branch.setAttribute('direction', direction)
    return branch
    
def print_a_document(doc):
    for e in doc.childNodes:
        if e.localName == 'root':
            print e.localName
        elif e.localName == 'branch':
            print "- %s %s" % (e.getAttribute('direction'), e.localName)
        elif e.localName == 'leaf':
            print "-- %s %s" % (e.childNodes[0].toxml(), e.localName)
        print_a_document(e)
        

writer = dom.Document()
root = writer.createElement("root")

directions = ['east', 'north', 'west', 'south', 'northwest', 'northeast', 'southwest', 'southeast']
colors = ['gray', 'red', 'green', 'blue', 'yellow', 'golden', 'rusty']

for i in range(3):
    grow = random.randint(0, 7)
    branch = create_branch(writer, directions[grow])
    for j in range(3):
        grow = random.randint(0, 6)
        leaf = create_leaf(writer, colors[grow])
        branch.appendChild(leaf)
    root.appendChild(branch)
    
print 'writing xml:\n', root.toxml()

writer.appendChild(root)
file_out = open('./foo.xml', "w")
writer.writexml(file_out)
file_out.close()

print 'read back'

file_in = open('./foo.xml', "r")
reader = dom.parse(file_in)
file_in.close()

print_a_document(reader)
