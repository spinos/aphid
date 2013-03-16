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
	union {
			/* Inner node */
			struct {
				/* Bit layout:
				   31   : False (inner node)
				   30   : Indirection node flag
				   29-3 : Offset to the left child 
				          or indirection table entry
				   2-0  : Split axis
				*/
				unsigned long combined;

				/// Split plane coordinate
				float split;
			} inner;

			/* Leaf node */
			struct {
				/* Bit layout:
				   31   : True (leaf node)
				   30-0 : Offset to the node's primitive list
				*/
				unsigned long combined;

				/// End offset of the primitive list
				unsigned long end;
			} leaf;
		};
		
	KdTreeNode();
	
	void initLeafNode(unsigned int offset, unsigned numPrims);
	void initInnerNode(int axis, float splitAt);
	
	void setSplitPos(float a_Pos );
	float getSplitPos() const;
	void setAxis( int a_Axis );
	int getAxis() const;
	void setLeaf( bool a_Leaf );
	unsigned long getPrimStart() const;
	unsigned long getNumPrims() const;
	bool isLeaf() const;
	void setLeft( KdTreeNode* a_Left );
	void setPrimStart(unsigned long offset);
	void setNumPrims(unsigned long numPrims);
	KdTreeNode* getLeft() const;
	KdTreeNode* getRight() const;

	enum EMask {
		EInnerAxisMask = ~0x3,
		ETypeMask = ~0x4,
		EIndirectionMask = 0x7,
		ELeafOffsetMask = ~ETypeMask,
		//EInnerOffsetMask = ~(EInnerAxisMask + EIndirectionMask),
		//ERelOffsetLimit = (1<<28) - 1
	};
	
private:
	//float m_Split;
	//unsigned long m_combined;
};