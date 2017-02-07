#ifndef HESPERISFILE_H
#define HESPERISFILE_H

/*
 *  HesperisFile.h
 *  hesperis
 *
 *  Created by jian zhang on 4/21/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <h5/HFile.h>
#include <h5/HBase.h>
#include <h5/HTransform.h>
#include <h5/HFrameRange.h>
#include <h5/HTriangleMeshGroup.h>
#include <h5/HTetrahedronMeshGroup.h>
#include <h5/HCurveGroup.h>
#include <h5/HWorld.h>
#include <h5/HPolygonalMesh.h>
#include <h5/HAttributeGroup.h>
#include <h5/HNumericBundle.h>
#include <string>
#include <map>
#include <vector>
#include <AFrameRange.h>

namespace aphid {
    
class BaseTransform;
class CurveGroup;
class BaseBuffer;
class ATetrahedronMeshGroup;
class ATriangleMesh;
class ATriangleMeshGroup;
class GeometryArray;
class APolygonalMesh;
class AAttribute;

class HesperisFile : public HFile {
    
/// 1-D array of data stored to path
    const ABundleAttribute * m_bundle;
    std::string m_bundlePath;
    
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
        WPoly = 4,
		WAttrib = 5,
	    WBundle = 6
	};
	
	HesperisFile();
	HesperisFile(const char * name);
	virtual ~HesperisFile();
	
	void setReadComponent(ReadComponent comp);
	void setWriteComponent(WriteComponent comp);
	
	void setBundleEntry(const ABundleAttribute * d,
	                    const std::string & parentName);
	void addAttribute(const std::string & name, AAttribute * data);
    void addTransform(const std::string & name, BaseTransform * data);
	void addCurve(const std::string & name, CurveGroup * data);
	void addTetrahedron(const std::string & name, ATetrahedronMeshGroup * data);
	void addTriangleMesh(const std::string & name, ATriangleMeshGroup * data);
	void addPolygonalMesh(const std::string & name, APolygonalMesh * data);
    
    virtual bool doWrite(const std::string & fileName);
	virtual bool doRead(const std::string & fileName);
	void extractCurves(GeometryArray * dst);
    void extractTetrahedronMeshes(GeometryArray * dst);
	void extractTriangleMeshes(GeometryArray * dst);

	static AFrameRange Frames;
	static bool DoStripNamespace;
    
    void clearTransforms();
    void clearCurves();
    void clearTriangleMeshes();
	void clearPolygonalMeshes();
    void clearTetrahedronMeshes();
	void clearAttributes();
	
	std::string modifiedTime();
    
    template<typename Th>
    static bool LsNames(std::vector<std::string> & dst, HBase * parent)
    {
        std::vector<std::string > tmNames;
        parent->lsTypedChild<HTransform>(tmNames);
        std::vector<std::string>::const_iterator ita = tmNames.begin();
        
        for(;ita!=tmNames.end();++ita) {
            HBase child(*ita);
            LsNames<Th>(dst, &child);
            child.close();
        }
        
        std::vector<std::string > crvNames;
        parent->lsTypedChild<Th>(crvNames);
        std::vector<std::string>::const_iterator itb = crvNames.begin();
        
        for(;itb!=crvNames.end();++itb)
            dst.push_back(*itb);
        
        return true;   
    }
    
    static std::string WorldPath(const std::string & name);
/// rm world in path
    static std::string LocalPath(const std::string & name);
    
protected: 

private: 
	bool writeFrames();
	bool writeTransform();
	bool writeCurve();
	bool writeTetrahedron();
	bool writeTriangle();
    bool writePolygon();
	bool writeAttribute();
	bool writeBundle();
    
	bool readFrames(HBase * grp);
	bool listCurve(HBase * grp);
	bool readCurve();
    bool listTetrahedron(HBase * grp);
	bool readTetrahedron();
	bool listTriangle(HBase * grp);
	bool readTriangle();
    
	std::string checkPath(const std::string & name) const;
private:
	std::map<std::string, BaseTransform * > m_transforms;
	std::map<std::string, CurveGroup * > m_curves;
	std::map<std::string, ATetrahedronMeshGroup * > m_terahedrons;
	std::map<std::string, ATriangleMeshGroup * > m_triangleMeshes;
    std::map<std::string, APolygonalMesh * > m_polyMeshes;
	std::map<std::string, AAttribute * > m_attribs;
    ReadComponent m_readComp;
	WriteComponent m_writeComp;
};

}
#endif        //  #ifndef HESPERISFILE_H
