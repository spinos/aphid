/*
 *  UniformGrid.h
 *  proxyPaint
 *
 *  Created by jian zhang on 2/5/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include <CartesianGrid.h>
class KdIntersection;

class UniformGrid : public CartesianGrid
{
	sdb::CellHash * m_cellsToRefine;
	int m_maxLevel;
	
public:
    UniformGrid();
	virtual ~UniformGrid();
    
    virtual void create(KdIntersection * tree, int maxLevel);
    const int & maxLevel() const;
	
protected:
    virtual bool tagCellsToRefine(KdIntersection * tree);
	void refine(KdIntersection * tree);
	
private:
    void setCellToRefine(unsigned k, const sdb::CellValue * v,
                         int toRefine);
    
private:
    
};
