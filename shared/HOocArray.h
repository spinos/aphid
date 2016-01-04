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

/// BufSize is max num columns in-core

template <int DataRank, int NRows, int BufSize>
class HOocArray : public H2dDataset<DataRank, NRows> {
	
/// in-core buffer
	char * m_data;
/// num char per column
	int m_nbits;
/// in-core buffer usage
	int m_incoreSize;
/// total space usage
	int m_size;
	hid_t m_parentId;
	
public:
	HOocArray(const std::string & name) : H2dDataset<DataRank, NRows>(name)
	{
		m_nbits = H2dDataset<DataRank, NRows>::NumBitsPerCol();
		m_data = new char[m_nbits * BufSize];
		m_incoreSize = 0;
		m_size = 0;
	}
	
	virtual ~HOocArray() 
	{
		delete m_data;
	}
	
	void createStorage(hid_t parentId);
	void openStorage(hid_t parentId);
	
	void insert(char * d);
	void finishInsert();
	
	int size() const
	{ return m_size; }
	
	void read(char * d, int offset, int ncols);
	void readIncore(int offset, int ncols);
	
	char * incoreBuf() const;
	
protected:

private:
	void writeCurrentBuf(int ncols);
	
};

template <int DataRank, int NRows, int BufSize>
void HOocArray<DataRank, NRows, BufSize>::createStorage(hid_t parentId)
{
	if(H2dDataset<DataRank, NRows>::create(parentId) ) {
		H2dDataset<DataRank, NRows>::close();
		m_parentId = parentId;
	}
	else
		m_parentId = 0;
}

template <int DataRank, int NRows, int BufSize>
void HOocArray<DataRank, NRows, BufSize>::openStorage(hid_t parentId)
{
	if(H2dDataset<DataRank, NRows>::open(parentId) ) {
		m_parentId = parentId;
	}
	else
		m_parentId = 0;
}

template <int DataRank, int NRows, int BufSize>
void HOocArray<DataRank, NRows, BufSize>::insert(char * d) 
{
	memcpy( &m_data[m_nbits * m_incoreSize], d, m_nbits );
	m_incoreSize++;
	m_size++;
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
	part.start[0] = m_size - ncols;
	part.start[1] = 0;
	part.count[0] = ncols;
	part.count[1] = NRows;
	
	H2dDataset<DataRank, NRows>::open(m_parentId);
	H2dDataset<DataRank, NRows>::write(m_data, &part);
	H2dDataset<DataRank, NRows>::close();
}

template <int DataRank, int NRows, int BufSize>
void HOocArray<DataRank, NRows, BufSize>::read(char * d, int offset, int ncols)
{
	hdata::Select2DPart part;
	part.start[0] = offset;
	part.start[1] = 0;
	part.count[0] = ncols;
	part.count[1] = NRows;
	
	H2dDataset<DataRank, NRows>::open(m_parentId);
	H2dDataset<DataRank, NRows>::read(d, &part);
	H2dDataset<DataRank, NRows>::close();
}

template <int DataRank, int NRows, int BufSize>
void HOocArray<DataRank, NRows, BufSize>::readIncore(int offset, int ncols)
{ read(m_data, offset, ncols); }

template <int DataRank, int NRows, int BufSize>
char * HOocArray<DataRank, NRows, BufSize>::incoreBuf() const
{ return m_data; }
