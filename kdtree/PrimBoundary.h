#pragma once

#include <Boundary.h>
#include <GridClustering.h>

namespace aphid {
    
class PrimBoundary : public Boundary {

    GridClustering * m_grid;
	sdb::VectorArray<BoundingBox> m_primitiveBoxes;
    sdb::VectorArray<unsigned> m_indices;
    unsigned m_numPrims;
    
public:
    PrimBoundary();
    virtual ~PrimBoundary();
    
    float visitCost() const;
	const unsigned & numPrims() const;
	bool isEmpty() const;
	void compressPrimitives();
    
	const sdb::VectorArray<unsigned> & indices() const;
	const unsigned & indexAt(const unsigned & idx) const;
	
	bool isCompressed() const;
	void verbose() const;
	bool canEndSubdivide(const float & costOfDevivde) const;
	
protected:
	void clearPrimitive();
	void addPrimitive(const unsigned & idx);
	void addPrimitiveBox(const BoundingBox & b);
	GridClustering * grid();
	const sdb::VectorArray<BoundingBox> & primitiveBoxes() const;
	void uncompressGrid(const sdb::VectorArray<BoundingBox> & boxSrc);
	void createGrid(const float & x);
	void addCell(const sdb::Coord3 & x, GroupCell * c);
	void countPrimsInGrid();
	
private:

};

}
