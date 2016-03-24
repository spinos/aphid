#pragma once

#include <Boundary.h>
#include <GridClustering.h>

namespace aphid {
    
class PrimBoundary : public Boundary {

    sdb::VectorArray<BoundingBox> m_primitiveBoxes;
    sdb::VectorArray<unsigned> m_indices;
    GridClustering * m_grid;
	int m_numPrims;
    
public:
    PrimBoundary();
    virtual ~PrimBoundary();
    
    float visitCost() const;
	const int & numPrims() const;
	bool isEmpty() const;
	
	const sdb::VectorArray<unsigned> & indices() const;
	const sdb::VectorArray<BoundingBox> & primitiveBoxes() const;
	
	bool isCompressed() const;
	void verbose() const;
	bool canEndSubdivide(const float & costOfDivivde) const;
	GridClustering * grid();
	void createGrid(const float & x);
	void addCell(const sdb::Coord3 & x, GroupCell * c);
	void addPrimitive(const unsigned & idx);
	
protected:
	void countPrimsInGrid();
	void clearPrimitive();
	void compressPrimitives();
    void addPrimitiveBox(const BoundingBox & b);
	void uncompressGrid(const sdb::VectorArray<BoundingBox> & boxSrc);
	bool decompress(const sdb::VectorArray<BoundingBox> & boxSrc, 
		bool forced = false);
	
private:

};

}
