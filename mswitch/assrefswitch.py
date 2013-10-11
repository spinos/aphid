from maya import OpenMaya as om

def ListReferenceFiles():
    res = []
    om.MFileIO.getReferences(res)
    return res
    
def GetReferenceNodes(filename):
    l = om.MSelectionList()
    om.MFileIO.getReferenceNodes(filename, l)
    return l
    
def SelectionListIntersects(a, b):
    it = om.MItSelectionList(a)
    item = om.MDagPath()
    while not it.isDone():
        it.getDagPath(item)
        
        if b.hasItem(item):
            return True
        it.next()
        
    return False
    
def GetSelectedReference():
    l0 = om.MSelectionList()
    om.MGlobal.getActiveSelectionList(l0)
    if l0.length() < 1:
        print 'zero selection'
        return 'unknown'
        
    files = ListReferenceFiles()
    
    for afile in files:
        l1 = GetReferenceNodes(afile)
        if SelectionListIntersects(l0, l1):
            return afile
            
    return 'unknown'
    
def GetSelectedStandin():
    l0 = om.MSelectionList()
    om.MGlobal.getActiveSelectionList(l0)
    if l0.length() < 1:
        print 'zero selection'
        return ('unknown', item) 

    it = om.MItSelectionList(l0)
    item = om.MDagPath()
    while not it.isDone():
        it.getDagPath(item)
        item.extendToShape()
        if om.MFnDagNode(item).typeName() == 'aiStandIn':
            return (om.MFnDagNode(item).typeName(), item)
        it.next()
    return ('unknown', item)
    
def GetSelectedTransform():
    l0 = om.MSelectionList()
    om.MGlobal.getActiveSelectionList(l0)
    it = om.MItSelectionList(l0)
    item = om.MDagPath()
    it.getDagPath(item)
    ft = om.MFnDagNode(item.transform())
    
    a = om.MCommandResult()
    om.MGlobal.executeCommand('getAttr '+ft.findPlug('translate').name(), a)
    translate3 = om.MDoubleArray()
    a.getResult(translate3)
    print 'translate ', translate3
    
    om.MGlobal.executeCommand('getAttr '+ft.findPlug('rotate').name(), a)
    rotate3 = om.MDoubleArray()
    a.getResult(rotate3)
    print 'rotate ', rotate3
    
    om.MGlobal.executeCommand('getAttr '+ft.findPlug('scale').name(), a)
    scale3 = om.MDoubleArray()
    a.getResult(scale3)
    print 'scale ', scale3
    
    return (translate3, rotate3, scale3)
    
def SetTransform(path, t, r, s):
    print 'restore transform ', path.fullPathName()
    
    c = 'setAttr "'+ path.fullPathName() +'.translate" %f %f %f ' % (t[0], t[1], t[2])
    om.MGlobal.executeCommand(c)
    c = 'setAttr "'+ path.fullPathName() +'.rotate" %f %f %f ' % (r[0], r[1], r[2])
    om.MGlobal.executeCommand(c)
    c = 'setAttr "'+ path.fullPathName() +'.scale" %f %f %f ' % (s[0], s[1], s[2])
    om.MGlobal.executeCommand(c)
    
def SwitchFromReferenceToAss(assFile):
    refFile = GetSelectedReference()
    if refFile == 'unknown':
        print 'no reference selected, do nothing'
        return False
        
    (t, r, s) = GetSelectedTransform()
   
    print 'remove reference ', refFile
    om.MFileIO.removeReference(refFile)
    
    print 'standin ', assFile
    mod = om.MDagModifier()
    
    standin = mod.createNode('aiStandIn')
    mod.doIt()

    path = om.MDagPath.getAPathTo(standin)

    SetTransform(path, t, r, s)
    
    path.extendToShape()
    fstandin = om.MFnDependencyNode(path.node())
    
    print 'created ', fstandin.name()
    print 'set ', fstandin.findPlug('dso').name()
    
    c = 'setAttr  -type "string" "'+ fstandin.findPlug('dso').name() +'" "'+assFile+'"'
    om.MGlobal.executeCommand(c)
    return True
    
def SwitchFromAssToReference(refFileName):        
    (typeName, standin) = GetSelectedStandin()
    if typeName is 'unknown':
        print 'no standin selected, do nothing'
        return False
        
    (t, r, s) = GetSelectedTransform()
        
    om.MGlobal.executeCommand('delete')
    
    print 'reference ', refFileName
    om.MFileIO.reference(refFileName)
    
    l = GetReferenceNodes(refFileName)
    it = om.MItSelectionList(l)
    trans = om.MDagPath()
    it.getDagPath(trans)

    SetTransform(trans, t, r, s)
    
    return True
    
## SwitchFromReferenceToAss('D:/man/facialRigExample/face_valleyGirl_class/scenes/test.ass')
## SwitchFromAssToReference('D:/man/facialRigExample/face_valleyGirl_class/scenes/test.ma')
    
    
    
    
    


    


