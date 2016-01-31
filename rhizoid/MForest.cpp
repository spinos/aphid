#include "MForest.h"
#include <maya/MFnMesh.h>
#include <gl_heads.h>
#include <AHelper.h>

MForest::MForest()
{}

MForest::~MForest()
{}

void MForest::updateGround(MArrayDataHandle & data)
{
    unsigned nslots = data.elementCount();
    if(numGroundMeshes() > 0 && numGroundMeshes() != nslots)
        clearGroundMeshes();
    
    for(unsigned i=0;i<nslots;++i) {
        MDataHandle meshData = data.inputValue();
        MObject mesh = meshData.asMesh();
        if(mesh.isNull()) {
			AHelper::Info<unsigned>("MForest error no input ground mesh", i );
		}
        else {
            updateGroundMesh(mesh, i);
        }
        
        data.next();
    }
    
    if(numGroundMeshes() < 1) {
        AHelper::Info<int>("MForest no ground", 0);
        return;
    }
    
    buildGround();
}

void MForest::updateGroundMesh(MObject & mesh, unsigned idx)
{
    MFnMesh fmesh(mesh);
	
	MPointArray ps;
	fmesh.getPoints(ps);
	
	const unsigned nv = ps.length();
	unsigned i = 0;
	//for(;i<nv;i++) ps[i] *= wm;
	
	MIntArray triangleCounts, triangleVertices;
	fmesh.getTriangles(triangleCounts, triangleVertices);
	
    ATriangleMesh * trimesh = getGroundMesh(idx);
    bool toRebuild = false;
    if(!trimesh) {
        toRebuild = true;
        trimesh = new ATriangleMesh;
    }
    else {
        if(trimesh->numPoints() != nv || 
            trimesh->numTriangles() != triangleVertices.length()/3) {
            
            toRebuild = true;
        }
    }

    if(toRebuild) {
        trimesh->create(nv, triangleVertices.length()/3);
        unsigned * ind = trimesh->indices();
        for(i=0;i<triangleVertices.length();i++) ind[i] = triangleVertices[i];
	}
	
	Vector3F * cvs = trimesh->points();
	for(i=0;i<nv;i++) cvs[i].set(ps[i].x, ps[i].y, ps[i].z);
    
    if(toRebuild) {
        AHelper::Info<std::string>("MForest ground ", trimesh->verbosestr());
        setGroundMesh(trimesh, idx);
    }
}

void MForest::selectPlant(const MPoint & origin, const MPoint & dest, 
					MGlobal::ListAdjustment adj)
{
	Vector3F a(origin.x, origin.y, origin.z);
	Vector3F b(dest.x, dest.y, dest.z);
	Ray r(a, b);
	
	SelectionContext::SelectMode m = SelectionContext::Replace;
	if(adj == MGlobal::kAddToList) m = SelectionContext::Append;
	else if(adj == MGlobal::kRemoveFromList) m = SelectionContext::Remove;
	
	bool stat = selectPlants(r, m);
	if(!stat)
		AHelper::Info<int>("empty selection", m);
}

void MForest::selectGround(const MPoint & origin, const MPoint & dest, MGlobal::ListAdjustment adj)
{
	Vector3F a(origin.x, origin.y, origin.z);
	Vector3F b(dest.x, dest.y, dest.z);
	Ray r(a, b);
	
	SelectionContext::SelectMode m = SelectionContext::Replace;
	if(adj == MGlobal::kAddToList) m = SelectionContext::Append;
	else if(adj == MGlobal::kRemoveFromList) m = SelectionContext::Remove;
	
	bool stat = selectGroundFaces(r, m);
	if(!stat)
		AHelper::Info<int>("empty selection", m);
}

void MForest::flood(GrowOption & option)
{
	option.m_plantId = activePlantId();
	AHelper::Info<int>("ProxyViz begin flood plant", option.m_plantId);
	growOnGround(option);
	updateNumPlants();
	AHelper::Info<int>("ProxyViz end flood, result total plant count", numPlants() );
	updateGrid();
}

void MForest::grow(const MPoint & origin, const MPoint & dest, 
					GrowOption & option)
{
	option.m_plantId = activePlantId();
	Vector3F a(origin.x, origin.y, origin.z);
	Vector3F b(dest.x, dest.y, dest.z);
	Ray r(a, b);
	growAt(r, option);
}

void MForest::drawSolidMesh(MItMeshPolygon & iter)
{
	iter.reset();
	for(; !iter.isDone(); iter.next()) {
		int vertexCount = iter.polygonVertexCount();
		glBegin(GL_POLYGON);
		for(int i=0; i < vertexCount; i++) {
			MPoint p = iter.point (i);
			MVector n;
			iter.getNormal(i, n);
			glNormal3f(n.x, n.y, n.z);
			glVertex3f(p.x, p.y, p.z);
		}
		glEnd();
	}
}

void MForest::drawWireMesh(MItMeshPolygon & iter)
{
	iter.reset();
	glBegin(GL_LINES);
	for(; !iter.isDone(); iter.next()) {
		int vertexCount = iter.polygonVertexCount();
		
		for(int i=0; i < vertexCount-1; i++) {
			MPoint p = iter.point (i);
			glVertex3f(p.x, p.y, p.z);
			p = iter.point (i+1);
			glVertex3f(p.x, p.y, p.z);
		}		
	}
	glEnd();
}

void MForest::matrix_as_array(const MMatrix &space, double *mm)
{
	mm[0] = space(0,0);
	mm[1] = space(0,1);
	mm[2] = space(0,2);
	mm[3] = space(0,3);
	mm[4] = space(1,0);
	mm[5] = space(1,1);
	mm[6] = space(1,2);
	mm[7] = space(1,3);
	mm[8] = space(2,0);
	mm[9] = space(2,1);
	mm[10] = space(2,2);
	mm[11] = space(2,3);
	mm[12] = space(3,0);
	mm[13] = space(3,1);
	mm[14] = space(3,2);
	mm[15] = space(3,3);
}

void MForest::finishGrow()
{
	updateGrid();
	updateNumPlants();
}

void MForest::erase(const MPoint & origin, const MPoint & dest,
					float weight)
{
	Vector3F a(origin.x, origin.y, origin.z);
	Vector3F b(dest.x, dest.y, dest.z);
	Ray r(a, b);
	
	clearAt(r, weight);
}

void MForest::finishErase()
{
	updateGrid();
	updateNumPlants();
}
//:~