#ifndef CARTESIANGRID_H
#define CARTESIANGRID_H

#include <BoundingBox.h>
#include <MortonHash.h>
class BaseBuffer;
class CartesianGrid 
{
public:
    CartesianGrid();
	virtual ~CartesianGrid();
    
    void setBounding(float * originSpan);
	void setBounding(const BoundingBox & bound);
    
    const unsigned numCells() const;
    void getBounding(BoundingBox & bound) const;
    const Vector3F origin() const;
	const float span() const;
    
	sdb::CellHash * cells();
    const Vector3F cellCenter(unsigned code) const;
    const float cellSizeAtLevel(int level) const;
	BoundingBox cellBox(unsigned code, int level) const;
	
	void addCell(unsigned code, int level, int visited, unsigned index);
	
    unsigned mortonEncodeLevel(const Vector3F & p, int level) const;
	void printGrids(BaseBuffer * dst);
	
	bool isPInsideBound(const Vector3F & p) const;
	void putPInsideBound(Vector3F & p) const;
	
protected:
	const float gridSize() const;
	
	void gridOfP(const Vector3F & p, unsigned & x,
									unsigned & y,
									unsigned & z) const;
	void gridOfCell(unsigned & x,
									unsigned & y,
									unsigned & z,
									int level) const;
	const unsigned mortonEncode(const Vector3F & p) const;
	sdb::CellValue * findGrid(unsigned code) const;
	sdb::CellValue * findCell(unsigned code) const;
	unsigned addGrid(const Vector3F & p);
    unsigned addCell(const Vector3F & p, int level);
	const Vector3F gridOrigin(unsigned code) const;
	const Vector3F cellOrigin(unsigned code, int level) const;
	void removeCell(unsigned code);
    void printHash();
	
	unsigned encodeNeighborCell(unsigned code, 
									int level,
									int dx, int dy, int dz);

    sdb::CellValue * findNeighborCell(unsigned code, 
									int level,
									int dx, int dy, int dz);
									
	unsigned encodeFinerNeighborCell(unsigned code, 
									int level,
									int dx, int dy, int dz,
									int cx, int cy, int cz) const;
									
	sdb::CellValue * findFinerNeighborCell(unsigned code, 
									int level,
									int dx, int dy, int dz,
									int cx, int cy, int cz);
private:
    Vector3F m_origin;
    float m_span, m_gridH; // same for each dimensions
    sdb::CellHash * m_cellHash;
    unsigned m_numCells;
};

#endif        //  #ifndef CARTESIANGRID_H

