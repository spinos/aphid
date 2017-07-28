from maya import OpenMaya as om

def GetSelectedBones(dst):
    mg = om.MGlobal()
    sel = om.MSelectionList()
    mg.getActiveSelectionList(sel)
    print(' %i selected' % sel.length())
    if sel.isEmpty() :
        return False
        
    for it in range(0, sel.length() ) :
        onem = om.MDagPath()
        sel.getDagPath(it, onem)
        
        if onem.apiType() == om.MFn.kJoint:
            print('add bone %s to selection ' % onem.fullPathName() )
            dst.append(onem)
            
    print(' selected %i joints ' % dst.length())
    return dst.length() > 0
    
def getWorldUp(boneDag):
    fbone = om.MFnTransform(boneDag)
    boneWq = om.MQuaternion()
    fbone.getRotation(boneWq, om.MSpace.kWorld)
    tm = om.MTransformationMatrix()
    tm.setRotationQuaternion(boneWq.x, boneWq.y, boneWq.z, boneWq.w)
    mat = tm.asMatrix()
    upv = om.MVector(0,1,0)
    upv = upv.transformAsNormal(mat)
    return upv
    
def writeAGhostPoint(rec, prof, boneDag, childDag):
    midUp = getWorldUp(boneDag)
    midUp.normalize()
    
    fbone = om.MFnTransform(boneDag)
    boneX = fbone.getTranslation(om.MSpace.kWorld)
    
    fchild = om.MFnTransform(childDag)
    childX = fchild.getTranslation(om.MSpace.kWorld)
    midP = (boneX + childX) * 0.5
    gp = midP + midUp
    rec.write("\n{%ff, %ff, %ff}," % (gp.x, gp.y, gp.z))
    
def writeAnEdgeConstraint(rec, prof, boneDag, childDag):
    iA = prof.nodeMap[boneDag.fullPathName()]
    iB = prof.nodeMap[childDag.fullPathName()]
    rec.write("\n{%i, %i, %i}," % (iA, iB, prof.ghostIt))
    iG = iA * 65536 + iB
    prof.ghostMap[iG] = prof.ghostIt
    
def writeABendTwistConstraint(rec, prof, boneDag, childDag, grandChildDag ):
    iA = prof.nodeMap[boneDag.fullPathName()]
    iB = prof.nodeMap[childDag.fullPathName()]
    iC = prof.nodeMap[grandChildDag.fullPathName()]
    d = iA * 65536 + iB
    iD = prof.ghostMap[d]
    e = iB * 65536 + iC
    iE = prof.ghostMap[e]
    rec.write("\n{%i, %i, %i, %i, %i}," % (iA, iB, iC, iD, iE))
    
def getNumChild(boneDag):
    fdag = om.MFnDagNode(boneDag)
    return fdag.childCount()
    
def getChildDag(boneDag, i):
    fdag = om.MFnDagNode(boneDag)
    ochild = fdag.child(i)
    childDag = om.MDagPath.getAPathTo(ochild)
    return childDag
    
def writeParticlePoints(rec, prof):
    c = 0
    boneIt = om.MItDag()
    boneIt.reset(prof.rootDag)
    while not boneIt.isDone() :
        boneDag = om.MDagPath()
        boneIt.getPath(boneDag)
        prof.nodeMap[boneDag.fullPathName()] = c
        c = c + 1
        boneIt.next()
    
    np = len(prof.nodeMap)
    print(' write %i particles ' % np )
    rec.write('\nstatic const int s%sNumParticles = %i;' % (prof.prefixName, np) )
    rec.write('\nstatic const float s%sParticlePoints[%i][3] = {' % (prof.prefixName, np) )
    
    ng = 0
    boneIt.reset(prof.rootDag)
    while not boneIt.isDone() :
        boneDag = om.MDagPath()
        boneIt.getPath(boneDag)
        fbone = om.MFnTransform(boneDag)
        wp = fbone.getTranslation(om.MSpace.kWorld)
        rec.write("\n{%ff, %ff, %ff}," % (wp.x, wp.y, wp.z))
        
        fdag = om.MFnDagNode(boneDag)
        if fdag.childCount() > 0 :
            ng = ng + fdag.childCount()
        boneIt.next()
        
    rec.write('\n};')
    
    print(' write %i ghost particles ' % ng )
    rec.write('\nstatic const int s%sNumGhostParticles = %i;' % (prof.prefixName, ng) )
    rec.write('\nstatic const float s%sGhostParticlePoints[%i][3] = {' % (prof.prefixName, ng) )
    
    prof.ghostIt = 0
    boneIt.reset(prof.rootDag)
    while not boneIt.isDone() :
        boneDag = om.MDagPath()
        boneIt.getPath(boneDag)
         
        fdag = om.MFnDagNode(boneDag)
        for i in range(0, fdag.childCount() ) :
            ochild = fdag.child(i)
            childDag = om.MDagPath.getAPathTo(ochild)
            writeAGhostPoint(rec, prof, boneDag, childDag)
            prof.ghostIt = prof.ghostIt + 1
            
        boneIt.next()
        
    rec.write('\n};')
    
    print(' write %i edge constraint ' % ng )
    rec.write('\nstatic const int s%sNumEdgeConstraints = %i;' % (prof.prefixName, ng) )
    rec.write('\nstatic const int s%sEdgeConstraints[%i][3] = {' % (prof.prefixName, ng) )
    
    prof.ghostIt = 0
    boneIt.reset(prof.rootDag)
    while not boneIt.isDone() :
        boneDag = om.MDagPath()
        boneIt.getPath(boneDag)
         
        fdag = om.MFnDagNode(boneDag)
        for i in range(0, fdag.childCount() ) :
            ochild = fdag.child(i)
            childDag = om.MDagPath.getAPathTo(ochild)
            writeAnEdgeConstraint(rec, prof, boneDag, childDag)
            prof.ghostIt = prof.ghostIt + 1
            
        boneIt.next()
        
    rec.write('\n};')
    
    nbtc = 0
    boneIt.reset(prof.rootDag)
    while not boneIt.isDone() :
        boneDag = om.MDagPath()
        boneIt.getPath(boneDag)
         
        fdag = om.MFnDagNode(boneDag)
        for i in range(0, fdag.childCount() ) :
            ochild = fdag.child(i)
            childDag = om.MDagPath.getAPathTo(ochild)
            nbtc = nbtc + getNumChild(childDag)
         
        boneIt.next()
    
    print(' write %i bend-twist constraint ' % nbtc )
    rec.write('\nstatic const int s%sNumBendAndTwistConstraints = %i;' % (prof.prefixName, nbtc) )
    rec.write('\nstatic const int s%sBendAndTwistConstraints[%i][5] = {' % (prof.prefixName, nbtc) )
    
    boneIt.reset(prof.rootDag)
    while not boneIt.isDone() :
        boneDag = om.MDagPath()
        boneIt.getPath(boneDag)
         
        fdag = om.MFnDagNode(boneDag)
        for i in range(0, fdag.childCount() ) :
            ochild = fdag.child(i)
            childDag = om.MDagPath.getAPathTo(ochild)
            for j in range(0, getNumChild(childDag)):
                writeABendTwistConstraint(rec, prof, boneDag, childDag, getChildDag(childDag, j) )
            
        boneIt.next()
        
    rec.write('\n};')
    
def CatchSelectedBones(fileName, prof):
    bones = om.MDagPathArray()
    if not GetSelectedBones(bones):
        return False
        
    rec = open(fileName, 'w')
    
    prof.rootDag = bones[0];
    writeParticlePoints(rec, prof);
    
    rec.close()
    print('catched mesh into file: %s' % fileName)
    return True
    
class CatBoneFrofile():
    def __init__(self, prefix):
        self.prefixName = prefix
        self.rootDag = None
        self.nodeMap = {}
        self.ghostMap = {}
        self.ghostIt = 0
   
prof = CatBoneFrofile('Foo')

CatchSelectedBones('D:/aphid/erod/rod/bones.h', prof)

