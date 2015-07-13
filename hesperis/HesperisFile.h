/*
 *  HesperisFile.h
 *  hesperis
 *
 *  Created by jian zhang on 4/21/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <HFile.h>
#include <string>
#include <map>
#include <vector>
#include <AFrameRange.h>
class BaseTransform;
class CurveGroup;
class BaseBuffer;
class ATetrahedronMesh;
class ATriangleMesh;
class ATriangleMeshGroup;
class HBase;
class GeometryArray;
class APolygonalMesh;

class HesperisFile : public HFile {
public:
	enum ReadComponent {
		RNone = 0,
		RCurve = 1,
		RTetra = 2,
		RTri = 3,
        RTransform = 4,
        RPoly = 5
	};

	enum WriteComponent {
		WCurve = 0,
		WTetra = 1,
		WTri = 2,
		WTransform = 3,
        WPoly = 4
	};
	
	HesperisFile();
	HesperisFile(const char * name);
	virtual ~HesperisFile();
	
	void setReadComponent(ReadComponent comp);
	void setWriteComponent(WriteComponent comp);
	
    void addTransform(const std::string & name, BaseTransform * data);
	void addCurve(const std::string & name, CurveGroup * data);
	void addTetrahedron(const std::string & name, ATetrahedronMesh * data);
	void addTriangleMesh(const std::string & name, ATriangleMeshGroup * data);
	void addPolygonalMesh(const std::string & name, APolygonalMesh * data);
    
    virtual bool doWrite(const std::string & fileName);
	virtual bool doRead(const std::string & fileName);
    void extractTetrahedronMeshes(GeometryArray * dst);
	void extractTriangleMeshes(GeometryArray * dst);
    
    void clearTransforms();
    void clearCurves();
    void clearTriangleMeshes();
	void clearPolygonalMeshes();
    
	static AFrameRange Frames;
	static bool DoStripNamespace;
protected:

private:
	bool writeFrames();
	bool writeTransform();
	bool writeCurve();
	bool writeTetrahedron();
	bool writeTriangle();
    bool writePolygon();
    
	bool readFrames(HBase * grp);
	bool readCurve();
    bool listTetrahedron(HBase * grp);
	bool readTetrahedron();
	bool listTriangle(HBase * grp);
	bool readTriangle();
    std::string worldPath(const std::string & name) const;
	std::string checkPath(const std::string & name) const;
private:
	std::map<std::string, BaseTransform * > m_transforms;
	std::map<std::string, CurveGroup * > m_curves;
	std::map<std::string, ATetrahedronMesh * > m_terahedrons;
	std::map<std::string, ATriangleMeshGroup * > m_triangleMeshes;
    std::map<std::string, APolygonalMesh * > m_polyMeshes;
    ReadComponent m_readComp;
	WriteComponent m_writeComp;
};