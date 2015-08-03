#include <CartesianGrid.h>
class BaseBuffer;
class KdIntersection;
class AdaptiveGrid : public CartesianGrid
{
public:
    AdaptiveGrid(const BoundingBox & bound);
    virtual ~AdaptiveGrid();
    
    void create(KdIntersection * tree, int maxLevel = 6);
protected:
    virtual bool tagCellsToRefine(KdIntersection * tree);
    void refine(KdIntersection * tree);
    
private:
    void setCellToRefine(unsigned k, const sdb::CellValue * v,
                         int toRefine);
    bool cellNeedRefine(unsigned k);
    bool check24NeighboursToRefine(unsigned k, const sdb::CellValue * v);
    bool multipleChildrenTouched(KdIntersection * tree,
                                 const Vector3F & parentCenter,
                                 float parentSize);
private:
    sdb::CellHash * m_cellsToRefine;
};
