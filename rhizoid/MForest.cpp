#include "MForest.h"
#include <maya/MFnMesh.h>
#include <gl_heads.h>
#include <AHelper.h>
#include <fstream> 

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
	if(!stat) AHelper::Info<int>("MForest error empty selection", 0);
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
	if(!stat) AHelper::Info<int>("MForest error empty ground", 0);
}

void MForest::flood(GrowOption & option)
{
	option.m_plantId = activePlantId();
	AHelper::Info<int>("ProxyViz begin flood plant", option.m_plantId);
	if(!growOnGround(option))
        AHelper::Info<int>("MForest error empty ground", 0 );
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

void MForest::savePlants(MPointArray & plantTms, 
					MIntArray & plantIds,
					MIntArray & plantTris,
					MVectorArray & plantCoords)
{
	sdb::WorldGrid<sdb::Array<int, sdb::Plant>, sdb::Plant > * g = grid();
	g->begin();
	while(!g->end() ) {
		saveCell(g->value(), plantTms, plantIds,
					plantTris, plantCoords);
		g->next();
	}
	AHelper::Info<unsigned>(" MForest saved num plants", plantIds.length() );
}

void MForest::saveCell(sdb::Array<int, sdb::Plant> *cell,
					MPointArray & plantTms, 
					MIntArray & plantIds,
					MIntArray & plantTris,
					MVectorArray & plantCoords)
{
	cell->begin();
	while(!cell->end() ) {
		sdb::PlantData * d = cell->value()->index;
		Matrix44F * mat = d->t1;
		plantTms.append(MPoint(mat->M(0,0), mat->M(0,1), mat->M(0,2) ) );
		plantTms.append(MPoint(mat->M(1,0), mat->M(1,1), mat->M(1,2) ) );
		plantTms.append(MPoint(mat->M(2,0), mat->M(2,1), mat->M(2,2) ) );
		plantTms.append(MPoint(mat->M(3,0), mat->M(3,1), mat->M(3,2) ) );
		
		sdb::GroundBind * bind = d->t2;
		plantTris.append(bind->m_geomComp);
		plantCoords.append(MVector(bind->m_w0, bind->m_w1, bind->m_w2) );
		
		int * pltyp = d->t3;
		plantIds.append(*pltyp);
		
		cell->next();
	}
}

bool MForest::loadPlants(const MPointArray & plantTms, 
					const MIntArray & plantIds,
					const MIntArray & plantTris,
					const MVectorArray & plantCoords)
{
	const unsigned npl = plantIds.length();
	if(npl<1) return false;
	if(npl != plantTris.length() ) return false;
	if(npl != plantCoords.length() ) return false;
	const unsigned ntm = plantTms.length();
	if(ntm != npl * 4) return false;
	
	Matrix44F tms;
	sdb::GroundBind bind;
	int typId;
	unsigned i =0;
	for(;i<npl;++i) {
		const MPoint & c0 = plantTms[i*4];
		*tms.m(0, 0) = c0.x;
		*tms.m(0, 1) = c0.y;
		*tms.m(0, 2) = c0.z;
		const MPoint & c1 = plantTms[i*4+1];
		*tms.m(1, 0) = c1.x;
		*tms.m(1, 1) = c1.y;
		*tms.m(1, 2) = c1.z;
		const MPoint & c2 = plantTms[i*4+2];
		*tms.m(2, 0) = c2.x;
		*tms.m(2, 1) = c2.y;
		*tms.m(2, 2) = c2.z;
		const MPoint & c3 = plantTms[i*4+3];
		*tms.m(3, 0) = c3.x;
		*tms.m(3, 1) = c3.y;
		*tms.m(3, 2) = c3.z;
		typId = plantIds[i];
		bind.m_geomComp = plantTris[i];
		const MVector & crd = plantCoords[i];
		bind.m_w0 = crd.x;
		bind.m_w1 = crd.y;
		bind.m_w2 = crd.z;
		addPlant(tms, bind, typId);
	}
	finishGrow();
	AHelper::Info<unsigned>(" MForest load num plants", numPlants() );
	return true;
}

void MForest::loadExternal(const char* filename)
{
    MGlobal::displayInfo("MForest loading external ...");
	std::ifstream chFile;
	chFile.open(filename, std::ios_base::in | std::ios_base::binary);
	if(!chFile.is_open()) {
		AHelper::Info<const char *>("MForest error cannot open file: ", filename);
		return;
	}
	
	chFile.seekg (0, ios::end);

	if(chFile.tellg() < 4 + 4 * 16) {
		AHelper::Info<const char *>("MForest error empty file: ", filename);
		chFile.close();
		return;
	}
	
	chFile.seekg (0, ios::beg);
	int numRec;
	chFile.read((char*)&numRec, sizeof(int));
	AHelper::Info<int>("MForest read record count ", numRec);
	float *data = new float[numRec * 16];
	chFile.read((char*)data, sizeof(float) * numRec * 16);
	chFile.close();
	
	Matrix44F space;
	sdb::GroundBind bind;
	bind.m_geomComp = -1;
	int typId = 0;	
	for(int i=0; i < numRec; i++) {
		const int ii = i * 16;
		*space.m(0, 0) = data[ii];
		*space.m(0, 1) = data[ii+1];
		*space.m(0, 2) = data[ii+2];
		*space.m(1, 0) = data[ii+4];
		*space.m(1, 1) = data[ii+5];
		*space.m(1, 2) = data[ii+6];
		*space.m(2, 0) = data[ii+8];
		*space.m(2, 1) = data[ii+9];
		*space.m(2, 2) = data[ii+10];
		*space.m(3, 0) = data[ii+12];
		*space.m(3, 1) = data[ii+13];
		*space.m(3, 2) = data[ii+14];
		
		addPlant(space, bind, typId);
	}
	delete[] data;
	finishGrow();
	AHelper::Info<const char *>("MForest read cache from ", filename);
}

void MForest::saveExternal(const char* filename)
{
	std::ofstream chFile;
	chFile.open(filename, std::ios_base::out | std::ios_base::binary);
	if(!chFile.is_open()) {
		MGlobal::displayWarning(MString("proxy viz cannot open file: ") + filename);
		return;
	}
	int numRec = 1;//_spaces.length();
	MGlobal::displayInfo(MString("proxy viz write recond count ") + numRec);
	chFile.write((char*)&numRec, sizeof(int));
	float *data = new float[numRec * 16];
	for(int i=0; i < numRec; i++) {
		const MMatrix space;// = _spaces[i];
		const int ii = i * 16;
		data[ii] = space(0, 0);
		data[ii+1] = space(0, 1);
		data[ii+2] = space(0, 2);
		data[ii+4] = space(1, 0);
		data[ii+5] = space(1, 1);
		data[ii+6] = space(1, 2);
		data[ii+8] = space(2, 0);
		data[ii+9] = space(2, 1);
		data[ii+10] = space(2, 2);
		data[ii+12] = space(3, 0);
		data[ii+13] = space(3, 1);
		data[ii+14] = space(3, 2);
	}
	chFile.write((char*)data, sizeof(float) * numRec * 16);
	chFile.close();
	MGlobal::displayInfo(MString("Well done! Proxy saved to ") + filename);
	delete[] data;
}

void MForest::bakePass(const char* filename, const MVectorArray & position, const MVectorArray & scale, const MVectorArray & rotation)
{
	std::ofstream chFile;
	chFile.open(filename, std::ios_base::out | std::ios_base::binary);
	if(!chFile.is_open()) {
		MGlobal::displayWarning(MString("proxy viz cannot open file: ") + filename);
		return;
	}
	int numRec = position.length();
	MGlobal::displayInfo(MString("proxy viz write bake recond count ") + numRec);
	chFile.write((char*)&numRec, sizeof(int));
	float *data = new float[numRec * 9];
	for(int i=0; i < numRec; i++) {
		const int ii = i * 9;
		const MVector pos = position[i];
		const MVector scl = scale[i];
		const MVector rot = rotation[i];
		data[ii] = pos.x;
		data[ii+1] = pos.y;
		data[ii+2] = pos.z;
		data[ii+3] = scl.x;
		data[ii+4] = scl.y;
		data[ii+5] = scl.z;
		data[ii+6] = rot.x;
		data[ii+7] = rot.y;
		data[ii+8] = rot.z;
	}
	chFile.write((char*)data, sizeof(float) * numRec * 9);
	chFile.close();
	MGlobal::displayInfo(MString("Well done! Proxy pass saved to ") + filename);
	delete[] data;
}
//:~