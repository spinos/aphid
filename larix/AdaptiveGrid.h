#include <CartesianGrid.h>
class BaseBuffer;
class KdIntersection;
class AdaptiveGrid : public CartesianGrid
{
public:
    AdaptiveGrid(const BoundingBox & bound);
    virtual ~AdaptiveGrid();
    
    virtual void create(KdIntersection * tree, int maxLevel = 6);
protected:
    virtual bool tagCellsToRefine(KdIntersection * tree);
	void tagCellsToRefineByNeighbours();
    void refine(KdIntersection * tree);
	
	class CellNeighbourInds {
	public:
		CellNeighbourInds() { reset(); }
		static unsigned InvalidIndex;
		void reset() {
			int i=0;
			for(;i<24;i++)
				_value[i] = InvalidIndex;
		}
		unsigned * side(int i) { return &_value[i<<2]; }
		unsigned _value[24];
	};
	
	CellNeighbourInds * findNeighbourCells(unsigned code);
	
private:
    void setCellToRefine(unsigned k, const sdb::CellValue * v,
                         int toRefine);
    bool cellNeedRefine(unsigned k);
    bool check24NeighboursToRefine(unsigned k, const sdb::CellValue * v);
    bool multipleChildrenTouched(KdIntersection * tree,
                                 const Vector3F & parentCenter,
                                 float parentSize);
	Vector3F neighbourCellCenter(int i, const Vector3F & p) const;
	void findFinerNeighbourCells(CellNeighbourInds * dst, int side,
								const Vector3F & center);
	Vector3F finerNeighbourCellCenter(int i, int side, const Vector3F & p) const;
private:
    sdb::CellHash * m_cellsToRefine;
};
