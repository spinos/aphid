#ifndef CARTESIANGRID_H
#define CARTESIANGRID_H

#include <BoundingBox.h>

class CartesianGrid 
{
public:
    struct CellIndex {
        unsigned key;
        unsigned index;
    };
    
    CartesianGrid(const BoundingBox & bound, int maxLevel);
    virtual ~CartesianGrid();
    
    const unsigned numCells() const;
    void getBounding(BoundingBox & bound) const;
    const Vector3F origin() const;
    
protected:
    CellIndex * cells();
    const float cellSizeAtLevel(int level) const;
    void addCell(const Vector3F & p, int level);
    void setCell(unsigned i, const Vector3F & p, int level);
    const Vector3F cellCenter(unsigned i) const;
private:
    Vector3F m_origin;
    float m_span; // same for each dimensions
    CellIndex * m_cells;
    unsigned * m_levels;
    unsigned m_numCells, m_maxNumCells;
};

#endif        //  #ifndef CARTESIANGRID_H

