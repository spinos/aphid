from maya import OpenMaya as om
from maya import OpenMayaAnim as oam

def FindAnimPlug(prof):
    mg = om.MGlobal()
    sel = om.MSelectionList()
    mg.getActiveSelectionList(sel)
    print(' %i selected' % sel.length())
    if sel.isEmpty():
        return False
        
    amu = oam.MAnimUtil()
    plgs = om.MPlugArray()
    amu.findAnimatedPlugs (sel, plgs)
    if plgs.length() < 1 :
        print(' no animated plug')

    for i in range(0, plgs.length() ) :
        if plgs[i].partialName(False ) == prof.plugName:
            print(' %s is animated' % plgs[i].partialName(False ) )
            prof.animPlug = om.MPlug(plgs[i])
            
    if prof.animPlug.isNull() :
        prof.animPlug = om.MPlug(plgs[0])
        
    return True

def WriteAnimCurveValue(prof):
    if not FindAnimPlug(prof):
        return
    print(' %s is sampled' % prof.animPlug.name() )
    rec = open(prof.fileName, 'w')
    
    mtm = om.MTime()
    acon = oam.MAnimControl()
    for i in range(prof.minFrame, prof.maxFrame+1):
        for j in range(0, prof.samplePerFrame):
            frame = i + j * prof.sampleTimeStep
            mtm.setValue(frame)
            acon.setCurrentTime(mtm)

            fval = prof.animPlug.asFloat()
            print(' %f: %f' % (frame, fval) )
            rec.write("%f\n" % fval)
            
    rec.close()
    print(' saved sample time file %s' % prof.fileName )
            
class CatAnimFrofile():
    def __init__(self):
        self.minFrame = 1
        self.maxFrame = 25
        self.samplePerFrame = 2
        self.sampleTimeStep = 0.25
        self.plugName = 'tx'
        self.animPlug = om.MPlug()
        self.fileName = 'D:/foo.txt'

prof = CatAnimFrofile()
prof.maxFrame = 35
    
WriteAnimCurveValue(prof)
