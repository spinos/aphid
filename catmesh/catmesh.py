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
    
def GetSelectedMeshes(dst):
    mg = om.MGlobal()
    sel = om.MSelectionList()
    mg.getActiveSelectionList(sel)
    print(' %i selected' % sel.length())
    if sel.isEmpty():
        return False
        
    for it in range(0, sel.length() ) :
        onem = om.MDagPath()
        sel.getDagPath(it, onem)
        onem.extendToShape()
        if onem.apiType() == om.MFn.kMesh:
            print('add mesh %s ' % onem.fullPathName() )
            dst.append(onem)
        
    print(' selected %i meshes ' % dst.length())
    return dst.length() > 0
   
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
    
def WriteMeshVertexColor(fileRec, meshPath, prof) :
    print('write mesh vertex color %s' % meshPath.fullPathName())
    
    itvert = om.MItMeshVertex(meshPath)
    print('vertex count: %i' % itvert.count())
    
    nf = itvert.count() * 3
    
    fileRec.write('static const float s%s_%sVertexColors[%i] = {' % (prof.prefixName, meshPath.partialPathName(), nf) )
    
    count = 0
    while not itvert.isDone():
        col = om.MColor()
        itvert.getColor(col, prof.colorName )
        
        fileRec.write("%ff, %ff, %ff, " % (col.r, col.g, col.b) )
        
        count += 1
        if (count % 32 == 0) or (count == itvert.count() - 1):
            fileRec.write("\n")
            
        itvert.next()
            
    fileRec.write('};\n')
    
    
def CatchSelectedMeshVertexColor(fileName, prof):
    meshes = om.MDagPathArray()
    if not GetSelectedMeshes(meshes):
        return False
        
    rec = open(fileName, 'w')
    
    for it in range(0, meshes.length() ) :
        WriteMeshVertexColor(rec, meshes[it], prof)
      
    rec.close()
    print('catched mesh into file: %s' % fileName)
    return True
    
def WriteMeshTexcoordFV(fileRec, meshPath, prof) :
    print('write mesh face varying texcoord %s' % meshPath.fullPathName())
    
    itface = om.MItMeshPolygon(meshPath)
    print('polygon count: %i' % itface.count() )
    
    nt = 0
    uarr = om.MFloatArray()
    varr = om.MFloatArray()
    while not itface.isDone() :
        itface.getUVs(uarr, varr);
        nt += uarr.length() - 2
        
        itface.next()
    
    print('triangle count: %i' % nt )
    
    nf = nt * 3 * 2
    
    fileRec.write('static const float s%s_%sTriangleTexcoords[%i] = {' % (prof.prefixName, meshPath.partialPathName(), nf) )
    
    itface.reset()
    count = 0
    while not itface.isDone():
        itface.getUVs(uarr, varr);
        
        for ti in range(0, uarr.length() - 2 ) :
            fileRec.write("%ff, %ff, " % (uarr[0], varr[0] ) )
            fileRec.write("%ff, %ff, " % (uarr[ti + 1], varr[ti + 1] ) )
            fileRec.write("%ff, %ff, " % (uarr[ti + 2], varr[ti + 2] ) )
        
            count += 1
            if (count % 32 == 0) or (count == nt - 1):
                fileRec.write("\n")
            
        itface.next()
            
    fileRec.write('};\n')
    
def CatchSelectedMeshTexcoordFV(fileName, prof):
    meshes = om.MDagPathArray()
    if not GetSelectedMeshes(meshes):
        return False
        
    rec = open(fileName, 'w')
    
    for it in range(0, meshes.length() ) :
        WriteMeshTexcoordFV(rec, meshes[it], prof)
      
    rec.close()
    print('catched mesh into file: %s' % fileName)
    return True

class CatMeshFrofile():
    def __init__(self, prefix):
        self.prefixName = prefix
        self.colorName = 'colorSet1'
        self.uvName = 'map1'
   
prof = CatMeshFrofile('foo')
##CatchSelectedMeshVertexColor('/Users/jianzhang/aphid/data/foo.h', prof)
##CatchSelectedMeshFV('/Users/jianzhang/aphid/data/box.h', 'box')
CatchSelectedMeshTexcoordFV('/Users/jianzhang/aphid/data/foo.h', prof)

