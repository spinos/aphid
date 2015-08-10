#include <CartesianGrid.h>
class BaseBuffer;
class KdIntersection;
class AdaptiveGrid : public CartesianGrid
{
public:
	AdaptiveGrid(float * originSpan);
    AdaptiveGrid(const BoundingBox & bound);
    virtual ~AdaptiveGrid();
    
    virtual void create(KdIntersection * tree, int maxLevel = 7);
	int maxLevel() const;
    void setMaxLevel(int x);
	sdb::CellValue * locateCell(const Vector3F & p) const;
	
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
        
        bool hasSide(int i)
        { 
            unsigned * idx = side(i);
            return (idx[0] < InvalidIndex
                    || idx[1] < InvalidIndex
                    || idx[2] < InvalidIndex
                    || idx[3] < InvalidIndex);
        }
        
        static bool IsValidIndex(unsigned idx)
        { return idx < InvalidIndex; }
        
		unsigned * side(int i) { return &_value[i<<2]; }
        void verbose() const
        {
            std::cout<<" cell neighbour ind "
            <<_value[0]<<","<<_value[1]<<","<<_value[2]<<","<<_value[3]<<";\n"
            <<_value[4]<<","<<_value[5]<<","<<_value[6]<<","<<_value[7]<<";\n"
            <<_value[8]<<","<<_value[9]<<","<<_value[10]<<","<<_value[11]<<";\n"
            <<_value[12]<<","<<_value[13]<<","<<_value[14]<<","<<_value[15]<<";\n"
            <<_value[16]<<","<<_value[17]<<","<<_value[18]<<","<<_value[19]<<";\n"
            <<_value[20]<<","<<_value[21]<<","<<_value[22]<<","<<_value[23]<<";\n";
        }
		
		int countSide(int i)
		{
			unsigned * s = side(i);
			int r = 0;
			if(s[0] < InvalidIndex) r++;
			if(s[1] < InvalidIndex) r++;
			if(s[2] < InvalidIndex) r++;
			if(s[3] < InvalidIndex) r++;
			return r;
		}
		
        unsigned _value[24];
	};
	
	void findNeighbourCells(CellNeighbourInds * dst, unsigned code,
                            sdb::CellValue * v);
	
private:
    void setCellToRefine(unsigned k, const sdb::CellValue * v,
                         int toRefine);
    bool cellNeedRefine(unsigned k);
    bool check24NeighboursToRefine(unsigned k, const sdb::CellValue * v);
    bool multipleChildrenTouched(KdIntersection * tree,
                                 const Vector3F & parentCenter,
                                 float parentSize);
	Vector3F neighbourCellCenter(int side, const Vector3F & p, float size) const;
	void findFinerNeighbourCells(CellNeighbourInds * dst, int side,
								const Vector3F & center, float size);
	Vector3F finerNeighbourCellCenter(int i, int side, const Vector3F & p, float size) const;
private:
    sdb::CellHash * m_cellsToRefine;
	int m_maxLevel;
};
