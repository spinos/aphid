/*
 *  HOocArray.h
 *  julia
 *
 *  Created by jian zhang on 1/3/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 *  out-of-core 2d array
 */

#pragma once
#include "H2dDataset.h"

namespace aphid {
    
/// BufSize is max num columns in-core

template <int DataRank, int NRows, int BufSize>
class HOocArray : public H2dDataset<DataRank, NRows, BufSize > {
	
/// in-core buffer
	char * m_data;
/// in-core buffer usage in ncols
	int m_incoreSize;
/// total space usage in nrows x ncols
	int m_pnts;
	hid_t m_parentId;
	int m_currentBlock;
	
public:
	HOocArray(const std::string & name);
	virtual ~HOocArray();
	
	bool createStorage(hid_t parentId);
	bool openStorage(hid_t parentId, bool doClear=false);
	
	void insert(char * d);
	void finishInsert();
	
	int numCols() const;
	const int & numPoints() const;
	
	void read(char * d, int offset, int ncols);
	void readIncore(int offset, int ncols);
	void readPoint(char * dst, int idx);
	void readColumn(char * dst, int idx);
	
	char * incoreBuf() const;
	
	void printValues();
/// set to beginning or somewhere
	void reset(const int & idx = 0);
	
protected:

private:
	void writeCurrentBuf(int ncols);
	void printIncoreValues(int &it, int ncols, int bpp);
	void printAValue(char * v);
};

template <int DataRank, int NRows, int BufSize>
HOocArray<DataRank, NRows, BufSize>::HOocArray(const std::string & name) : 
H2dDataset<DataRank, NRows, BufSize>(name)
{
    const int bpc = H2dDataset<DataRank, NRows, BufSize>::NumBitsPerCol();
    m_data = new char[bpc * BufSize];
    m_incoreSize = 0;
    m_pnts = 0;
}
	
template <int DataRank, int NRows, int BufSize>
HOocArray<DataRank, NRows, BufSize>::~HOocArray() 
{ delete m_data; }

template <int DataRank, int NRows, int BufSize>
bool HOocArray<DataRank, NRows, BufSize>::createStorage(hid_t parentId)
{
	if(H2dDataset<DataRank, NRows, BufSize>::create(parentId) ) {
		m_parentId = parentId;
		m_currentBlock = -1;
		H2dDataset<DataRank, NRows, BufSize>::close();
	}
	else
		m_parentId = 0;
	return m_parentId > 0;
}

template <int DataRank, int NRows, int BufSize>
bool HOocArray<DataRank, NRows, BufSize>::openStorage(hid_t parentId, bool doClear)
{
	if(H2dDataset<DataRank, NRows, BufSize>::open(parentId) ) {
		m_parentId = parentId;
		m_currentBlock = -1;
		m_pnts = H2dDataset<DataRank, NRows, BufSize>::checkDataSpace();
		if(m_pnts < 1) std::cout<<"\n HOocArray error: zero data space size";
		if(doClear) reset();
		H2dDataset<DataRank, NRows, BufSize>::close();
	}
	else
		m_parentId = 0;
	return m_parentId > 0;
}

template <int DataRank, int NRows, int BufSize>
void HOocArray<DataRank, NRows, BufSize>::insert(char * d) 
{
	int bpc = H2dDataset<DataRank, NRows, BufSize>::NumBitsPerCol();
		
	memcpy( &m_data[bpc * m_incoreSize], d, bpc );
	m_incoreSize++;
	m_pnts += NRows;
	if(m_incoreSize == BufSize) {
		writeCurrentBuf(m_incoreSize);
		m_incoreSize = 0;
	}
}

template <int DataRank, int NRows, int BufSize>
void HOocArray<DataRank, NRows, BufSize>::finishInsert()
{
	if(m_incoreSize == 0) return;
	
	writeCurrentBuf(m_incoreSize);
	m_incoreSize = 0;
}

template <int DataRank, int NRows, int BufSize>
void HOocArray<DataRank, NRows, BufSize>::writeCurrentBuf(int ncols)
{
	hdata::Select2DPart part;
	part.start[0] = numCols() - ncols;
	part.start[1] = 0;
	part.count[0] = ncols;
	part.count[1] = NRows;
	
	H2dDataset<DataRank, NRows, BufSize>::open(m_parentId);
	H2dDataset<DataRank, NRows, BufSize>::write(m_data, &part);
	H2dDataset<DataRank, NRows, BufSize>::close();
}

template <int DataRank, int NRows, int BufSize>
void HOocArray<DataRank, NRows, BufSize>::read(char * d, int offset, int ncols)
{
	hdata::Select2DPart part;
	part.start[0] = offset;
	part.start[1] = 0;
	part.count[0] = ncols;
	part.count[1] = NRows;
	
	H2dDataset<DataRank, NRows, BufSize>::open(m_parentId);
	H2dDataset<DataRank, NRows, BufSize>::read(d, &part);
	H2dDataset<DataRank, NRows, BufSize>::close();
}

template <int DataRank, int NRows, int BufSize>
void HOocArray<DataRank, NRows, BufSize>::readIncore(int offset, int ncols)
{ read(m_data, offset, ncols); }

template <int DataRank, int NRows, int BufSize>
char * HOocArray<DataRank, NRows, BufSize>::incoreBuf() const
{ return m_data; }

template <int DataRank, int NRows, int BufSize>
void HOocArray<DataRank, NRows, BufSize>::printValues()
{
    const int totalNCols = numCols();
	std::cout<<"\n n col "<<totalNCols;
	
	int numBlks = totalNCols / BufSize;
	if((totalNCols & (BufSize-1)) > 0) numBlks++;
	
	int ncols = BufSize;
	const int bpp = H2dDataset<DataRank, NRows, BufSize>::NumBitsPerPnt();
	int i, j, k, it = 0;
	for(i=0;i<numBlks;++i) {
		if((totalNCols - i*BufSize) < BufSize) ncols = totalNCols - i*BufSize;
		readIncore(i*BufSize, ncols);
		printIncoreValues(it, ncols, bpp);
	}
}

template <int DataRank, int NRows, int BufSize>
void HOocArray<DataRank, NRows, BufSize>::printIncoreValues(int &it, int ncols, int bpp)
{
	int j, k;
	for(j=0;j<ncols;++j) {
	    std::cout<<"\n ["<<it++<<"]";
		for(k=0;k<NRows;++k) {
			printAValue(&m_data[(j*NRows+k) * bpp]);
		}
	}
}

template <int DataRank, int NRows, int BufSize>
void HOocArray<DataRank, NRows, BufSize>::printAValue(char * v)
{
	switch (DataRank) {
		case hdata::TChar:
			std::cout<<" "<<*v;
			break;
		case hdata::TShort:
			std::cout<<" "<<*(short *)v;
			break;
		case hdata::TInt:
			std::cout<<" "<<*(int *)v;
			break;
		case hdata::TLongInt:
			std::cout<<" "<<*(long long *)v;
			break;
		case hdata::TFloat:
			std::cout<<" "<<*(float *)v;
			break;
		case hdata::TDouble:
			std::cout<<" "<<*(double *)v;
			break;
		default:
			break;
	}
}

template <int DataRank, int NRows, int BufSize>
void HOocArray<DataRank, NRows, BufSize>::readPoint(char * dst, int idx)
{
	if(idx >= m_pnts) {
		std::cout<<"\n HOocArray out-of-range index: "<<idx;
		return;
	}
	
	const int blk = idx / NRows / BufSize;
	const int blockOffset = blk * BufSize;
	const int bpp = H2dDataset<DataRank, NRows, BufSize>::NumBitsPerPnt();
	if(blk != m_currentBlock) {
		int ncols = (numCols() < blockOffset + BufSize) ? (numCols() - blockOffset) : BufSize; 
		readIncore(blockOffset, ncols);
		m_currentBlock = blk;
	}
	memcpy( dst, &m_data[bpp * (idx - blockOffset * NRows)], bpp );
}

template <int DataRank, int NRows, int BufSize>
void HOocArray<DataRank, NRows, BufSize>::readColumn(char * dst, int idx)
{
    if(idx >= numCols() ) {
		std::cout<<"\n HOocArray out-of-range column index: "<<idx;
		return;
	}
	
	const int blk = idx / BufSize;
	const int blockOffset = blk * BufSize;
	const int bpc = H2dDataset<DataRank, NRows, BufSize>::NumBitsPerCol();
	if(blk != m_currentBlock) {
		int ncols = (numCols() < blockOffset + BufSize) ? (numCols() - blockOffset) : BufSize; 
		readIncore(blockOffset, ncols);
		m_currentBlock = blk;
	}
	memcpy( dst, &m_data[bpc * (idx & (BufSize-1))], bpc );
}

template <int DataRank, int NRows, int BufSize>
void HOocArray<DataRank, NRows, BufSize>::reset(const int & idx)
{ m_pnts = idx; }

template <int DataRank, int NRows, int BufSize>
int HOocArray<DataRank, NRows, BufSize>::numCols() const
{ return m_pnts / NRows; }

template <int DataRank, int NRows, int BufSize>
const int & HOocArray<DataRank, NRows, BufSize>::numPoints() const
{ return m_pnts; }

}
//:~