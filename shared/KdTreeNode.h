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
	
	void setSplitPos(float a_Pos );
	float getSplitPos() const;
	void setAxis( int a_Axis );
	int getAxis() const;
	void setLeaf( bool a_Leaf );
	bool isLeaf() const;
	void setLeft( KdTreeNode* a_Left );
	KdTreeNode* getLeft() const;
	KdTreeNode* getRight() const;

	enum EMask {
		EInnerAxisMask = ~0x3,
		ETypeMask = ~0x4,
		EIndirectionMask = 0x7,
		//ELeafOffsetMask = ~ETypeMask,
		//EInnerOffsetMask = ~(EInnerAxisMask + EIndirectionMask),
		//ERelOffsetLimit = (1<<28) - 1
	};
	
private:
	float m_Split;
	unsigned long m_combined;
};