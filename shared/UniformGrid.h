/*
 *  UniformGrid.h
 *  proxyPaint
 *
 *  Created by jian zhang on 2/5/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include <CartesianGrid.h>
class KdTree;

class UniformGrid : public CartesianGrid
{
	sdb::CellHash * m_cellsToRefine;
	int m_maxLevel;
	
public:
    UniformGrid();
	virtual ~UniformGrid();
    
    virtual void create(KdTree * tree, int maxLevel);
    const int & maxLevel() const;
	
protected:
    virtual bool tagCellsToRefine(KdTree * tree);
	void refine(KdTree * tree);
	
private:
    void setCellToRefine(unsigned k, const sdb::CellValue * v,
                         int toRefine);
    
private:
    
};
