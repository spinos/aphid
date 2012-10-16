/*
 *  KdTreeNode.h
 *  kdtree
 *
 *  Created by jian zhang on 10/16/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

class KdTreeNode
{
public:
	KdTreeNode();
	void setAxis( int a_Axis );
	int getAxis() const;
	void setSplitPos(float a_Pos );
	float getSplitPos() const;
	void setLeft( KdTreeNode* a_Left );
	void setLeaf( bool a_Leaf );
	KdTreeNode* getLeft() const;
	KdTreeNode* getRight() const;
	bool isLeaf() const;
	
private:
	float m_Split;
	unsigned long m_Data;
};