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
   
def CatchSelectedMesh(fileName):
    mesh = om.MDagPath()
    if not GetSelectedMesh(mesh):
        return False
        
    print('selected mesh %s' % mesh.fullPathName())
    
    rec = open(fileName, 'w')
    
    fmesh = om.MFnMesh(mesh.node())
    print('vertex count: %i' % fmesh.numVertices())
    
    rec.write('static const int sNumVertices = %i;\n' % fmesh.numVertices())
    
    triangleCounts = om.MIntArray()
    triangleVertices = om.MIntArray()
    
    fmesh.getTriangles(triangleCounts, triangleVertices)
    print('triangle count: %i' % int(triangleVertices.length() / 3))
    
    rec.write('static const int sNumTriangleIndices = %i;\n' % triangleVertices.length())
    
    rec.write('static const int sMeshTriangleIndices[] = {')

    for i in range(0, triangleVertices.length() / 3):
        if i == (triangleVertices.length() / 3 - 1):
            rec.write("%i, %i, %i\n" % (triangleVertices[i * 3], triangleVertices[i * 3 + 1], triangleVertices[i * 3 + 2]))
        else:
            rec.write("%i, %i, %i,\n" % (triangleVertices[i * 3], triangleVertices[i * 3 + 1], triangleVertices[i * 3 + 2]))
        
    rec.write('};\n')
    
    vertexArray = om.MPointArray()
    fmesh.getPoints(vertexArray, om.MSpace.kObject)
    
    rec.write('static const float sMeshVertices[] = {')
    for i in range(0, vertexArray.length()):
        if i == (vertexArray.length() - 1):
            rec.write("%ff, %ff, %ff\n" % (vertexArray[i].x, vertexArray[i].y, vertexArray[i].z))
        else:
            rec.write("%ff, %ff, %ff,\n" % (vertexArray[i].x, vertexArray[i].y, vertexArray[i].z))
    rec.write('};\n')
    
    itv = om.MItMeshVertex(mesh);
    
    rec.write('static const float sMeshNormals[] = {')
    while not itv.isDone():
        nor = om.MVector()
        itv.getNormal(nor, om.MSpace.kObject)
        rec.write("%ff, %ff, %ff,\n" % (nor.x, nor.y, nor.z))
        itv.next()
    rec.write('};\n')    
    rec.close()
    print('catched mesh into file: %s' % fileName)
    return True
    

CatchSelectedMesh('D:/aphid/wheeled/Silverstone.h')
