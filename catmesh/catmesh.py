from maya import OpenMaya as om

def GetSelectedMesh(dst):
    mg = om.MGlobal()
    sel = om.MSelectionList()
    mg.getActiveSelectionList(sel)
    print(' %i selected' % sel.length())
    if sel.isEmpty():
        return False
        
    sel.getDagPath(0, dst)
    dst.extendToShape()
    if not dst.apiType() == om.MFn.kMesh:
        print('no mesh selected')
        return False
    return True
   
def CatchSelectedMesh(fileName, prefix):
    mesh = om.MDagPath()
    if not GetSelectedMesh(mesh):
        return False
        
    print('selected mesh %s' % mesh.fullPathName())
    
    rec = open(fileName, 'w')
    
    fmesh = om.MFnMesh(mesh.node())
    print('vertex count: %i' % fmesh.numVertices())
    
    rec.write('static const int s%sNumVertices = %i;\n' % (prefix, fmesh.numVertices()))
    
    triangleCounts = om.MIntArray()
    triangleVertices = om.MIntArray()
    
    fmesh.getTriangles(triangleCounts, triangleVertices)
    print('triangle count: %i' % int(triangleVertices.length() / 3))
    
    rec.write('static const int s%sNumTriangleIndices = %i;\n' % (prefix, triangleVertices.length()))
    
    rec.write('static const int s%sMeshTriangleIndices[] = {' % prefix)

    for i in range(0, triangleVertices.length() / 3):
        if i == (triangleVertices.length() / 3 - 1):
            rec.write("%i, %i, %i\n" % (triangleVertices[i * 3], triangleVertices[i * 3 + 1], triangleVertices[i * 3 + 2]))
        else:
            rec.write("%i, %i, %i,\n" % (triangleVertices[i * 3], triangleVertices[i * 3 + 1], triangleVertices[i * 3 + 2]))
        
    rec.write('};\n')
    
    vertexArray = om.MPointArray()
    fmesh.getPoints(vertexArray, om.MSpace.kObject)
    
    rec.write('static const float s%sMeshVertices[] = {' % prefix)
    for i in range(0, vertexArray.length()):
        if i == (vertexArray.length() - 1):
            rec.write("%ff, %ff, %ff\n" % (vertexArray[i].x, vertexArray[i].y, vertexArray[i].z))
        else:
            rec.write("%ff, %ff, %ff,\n" % (vertexArray[i].x, vertexArray[i].y, vertexArray[i].z))
    rec.write('};\n')
    
    itv = om.MItMeshVertex(mesh);
    
    rec.write('static const float s%sMeshNormals[] = {' % prefix)
    while not itv.isDone():
        nor = om.MVector()
        itv.getNormal(nor, om.MSpace.kObject)
        rec.write("%ff, %ff, %ff,\n" % (nor.x, nor.y, nor.z))
        itv.next()
    rec.write('};\n')    
    rec.close()
    print('catched mesh into file: %s' % fileName)
    return True
    
def writeTrianglePs(rec, trips):
    n = trips.length()
    i = 0;
    for i in range(0, n):
        ap = trips[i]
        rec.write("%ff, %ff, %ff,\n" % (ap.x, ap.y, ap.z))
        
        
def writeTriangleNs(rec, trips):
    n = trips.length()
    i = 0;
    for i in range(0, n/3):
        ap = trips[i*3]
        bp = trips[i*3+1]
        cp = trips[i*3+2]
        
        abv = bp - ap
        acv = cp - ap
        nv = abv ^ acv
        nv.normalize()
        
        rec.write("%ff, %ff, %ff,\n" % (nv.x, nv.y, nv.z))
        rec.write("%ff, %ff, %ff,\n" % (nv.x, nv.y, nv.z))
        rec.write("%ff, %ff, %ff,\n" % (nv.x, nv.y, nv.z))
        
   
def CatchSelectedMeshFV(fileName, prefix):
    mesh = om.MDagPath()
    if not GetSelectedMesh(mesh):
        return False
        
    print('selected mesh %s' % mesh.fullPathName())
    
    itpoly = om.MItMeshPolygon(mesh.node())
    print('face count: %i' % itpoly.count())
    
    ntriv = 0
    while not itpoly.isDone():
        trips = om.MPointArray()
        triind = om.MIntArray()
        itpoly.getTriangles(trips, triind)
        
        ntriv += triind.length()
        
        itpoly.next()
        
    print('triangle vertices count: %i' % ntriv )
    
    rec = open(fileName, 'w')
    
    rec.write('static const int s%sNumTriangleFVVertices = %i;\n' % (prefix, ntriv ) )
    rec.write('static const float s%sTriangleFVVertices[] = {' % prefix)
    
    itpoly.reset()
    while not itpoly.isDone():
        trips = om.MPointArray()
        triind = om.MIntArray()
        itpoly.getTriangles(trips, triind)
        
        writeTrianglePs(rec, trips)
        
        itpoly.next()
        
    rec.write('};\n')
    
    
    rec.write('static const float s%sTriangleFVNormals[] = {' % prefix)
    
    itpoly.reset()
    while not itpoly.isDone():
        trips = om.MPointArray()
        triind = om.MIntArray()
        itpoly.getTriangles(trips, triind)
        
        writeTriangleNs(rec, trips)
        
        itpoly.next()
        
    rec.write('};\n')
      
    rec.close()
    print('catched mesh into file: %s' % fileName)
    return True

CatchSelectedMeshFV('/Users/jianzhang/aphid/data/box.h', 'box')
