#ifndef APOLYGONALMESH_H
#define APOLYGONALMESH_H
#include <map>
#include "AGenericMesh.h"
class APolygonalUV;
class APolygonalMesh : public AGenericMesh {
public:
	APolygonalMesh();
	virtual ~APolygonalMesh();
	
	virtual const Type type() const;
	virtual const unsigned numComponents() const;
	virtual const BoundingBox calculateBBox(unsigned icomponent) const;
	const unsigned numPolygons() const;
	
	void create(unsigned np, unsigned ni, unsigned nf);
    void computeFaceDrift();
	unsigned faceCount(unsigned idx) const;
	unsigned * faceCounts() const;
    unsigned * faceDrifts() const;
    
	void addUV(const std::string & name, APolygonalUV * uv);
	const unsigned numUVs() const;
	const std::string uvName(unsigned idx) const;
	APolygonalUV * uvData(const std::string & name) const;
    
    virtual std::string verbosestr() const;
protected:
    unsigned * polygonIndices(unsigned idx) const;
private:
	std::map<std::string, APolygonalUV * > m_uvs;
    BaseBuffer * m_faceCounts;
    BaseBuffer * m_faceDrifts;
    unsigned m_numPolygons;
};
#endif        //  #ifndef APOLYGONALMESH_H

