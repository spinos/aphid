/*
 *  KdTreeNode.h
 *  kdtree
 *
 *  Created by jian zhang on 10/16/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
namespace aphid {

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
				int combined;

				/// Split plane coordinate
				float split;
			} inner;

			/* Leaf node */
			struct {
				/* Bit layout:
				   31   : True (leaf node)
				   30-0 : Offset to the node's primitive list
				*/
				int combined;

				/// End offset of the primitive list
				int end;
			} leaf;
		};
		
	KdTreeNode();
	
	void setSplitPos(float a_Pos );
	float getSplitPos() const;
	void setAxis( int a_Axis );
	int getAxis() const;
	void setLeaf();
    void setInternal();
	int getPrimStart() const;
	int getNumPrims() const;
	bool isLeaf() const;
	void setLeft( KdTreeNode* a_Left );
	void setPrimStart(unsigned long offset);
	void setNumPrims(unsigned long numPrims);
	void setOffset(int x);
	KdTreeNode* getLeft() const;
	KdTreeNode* getRight() const;
	int getOffset() const;

	enum EMask {
		EInnerAxisMask = ~0x3,
		ETypeMask = ~0x4,
        ETypeMaskTau = 0x4,
		ELeafOffsetMask = ~ETypeMask,
		EIndirectionMask = 0x7,
		
		//EInnerOffsetMask = ~(EInnerAxisMask + EIndirectionMask),
		//ERelOffsetLimit = (1<<28) - 1
	};
	
private:

};
}