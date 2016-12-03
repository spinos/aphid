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
			struct {
/// Bit layout:
///  -4  : offset to child
///   3  : false (inner node)
/// 2-0  : Split axis
				int combined;

/// split plane coordinate
				float split;
			} inner;

			struct {
/// Bit layout:
///  -4 : offset to first primitive
///   3 : true (leaf node)
/// 2-0 : no use
				int combined;

/// number of the primitives
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
		EInnerAxisMask = ~0x3,  // ...1100 
		ETypeMask = ~0x4,		// ...1011
        ETypeMaskTau = 0x4,		// ...0100
		EIndirectionMask = 0x7	// ....111 first 3 bits
	};
	
private:

};
}