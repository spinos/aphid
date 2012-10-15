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
	void SetAxis( int a_Axis );
	int GetAxis();
	void SetSplitPos(float a_Pos );
	float GetSplitPos();
	void SetLeft( KdTreeNode* a_Left );
	KdTreeNode* GetLeft();
	KdTreeNode* GetRight();
	bool IsLeaf();
	void SetLeaf( bool a_Leaf );
private:
	float m_Split;
	unsigned long m_Data;
};