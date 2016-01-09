/*
 *  H2dDataset.h
 *  hdata
 *
 *  Created by jian zhang on 1/3/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "HObject.h"

namespace hdata {

enum Rank {
	TUnknown = 0,
	TChar = 1,
	TShort = 2,
	TInt = 3,
	TLongInt = 4,
	TFloat = 5,
	TDouble = 6
};

struct Select2DPart {		
	hsize_t start[2];
	hsize_t count[2];
};

}

template<int DataRank, int NRows>
class H2dDataset : public HObject {

public:
	H2dDataset(const std::string & path);
	virtual ~H2dDataset();
	
	char open(hid_t parentId);
	char create(hid_t parentId);

	char write(char *data, hdata::Select2DPart * part);
	char read(char *data, hdata::Select2DPart * part);

	int numColumns();
	virtual int objectType() const;
	
	static int NumBitsPerPnt();
	static int NumBitsPerCol();
	
protected:
	
private:
	hid_t createMemSpace(hsize_t ncols) const;
	hid_t createMemSpace(hdata::Select2DPart * part) const;
	
	hid_t createFileSpace() const;
	hid_t dataType();
	
	char checkDataSpace();
};

template<int DataRank, int NRows>
H2dDataset<DataRank, NRows>::H2dDataset(const std::string & path) : HObject(path)
{}

template<int DataRank, int NRows>
H2dDataset<DataRank, NRows>::~H2dDataset() 
{}

template<int DataRank, int NRows>
char H2dDataset<DataRank, NRows>::open(hid_t parentId)
{
	fObjectId = H5Dopen(parentId, fObjectPath.c_str(), H5P_DEFAULT);
	
	if(fObjectId < 0)
		return 0;
		
	return 1;
}

template<int DataRank, int NRows>
char H2dDataset<DataRank, NRows>::checkDataSpace()
{
	hid_t fileSpace = H5Dget_space(fObjectId);
	if( H5Sget_simple_extent_ndims(fileSpace ) != 2) {
		std::cout<<"\nERROR: data space dimension != 2";
		return 0;
	}
	
	hsize_t dims[2];
	hsize_t maxdims[2];
	H5Sget_simple_extent_dims(fileSpace, dims,  maxdims );
	
	if(dims[1] != NRows) {
		std::cout<<"\nERROR: data space dims[1] != nrows";
		return 0;
	}
	return 1;
}

template<int DataRank, int NRows>
char H2dDataset<DataRank, NRows>::create(hid_t parentId)
{	
	if(open(parentId) ) {
		return checkDataSpace();
	}
	
	hid_t fileSpace = createFileSpace();
	
	if(fileSpace < 0) std::cout<<"\nh data space create failed\n";
	
	hid_t createProps = H5Pcreate(H5P_DATASET_CREATE);
	if(createProps < 0) std::cout<<"\nh create property failed\n";
	H5Pset_layout(createProps, H5D_CHUNKED);
	
	int ndim = 2;
	hsize_t chunkSize[2] = {1024, NRows};

	if(H5Pset_chunk(createProps, ndim, chunkSize)<0) {
      printf("\nError: fail to set chunk\n");
      return -1;
	}
	
	H5Pset_deflate(createProps, 8); 
   	
	fObjectId = H5Dcreate2(parentId, fObjectPath.c_str(), dataType(), fileSpace, 
                          H5P_DEFAULT, createProps, H5P_DEFAULT);
		  
	if(fObjectId < 0) {
	    std::cout<<"\nh data set create failed\n";
		return 0;
	}
	
	H5Sclose(fileSpace);
	H5Pclose(createProps);
	return 1;
}

template<int DataRank, int NRows>
hid_t H2dDataset<DataRank, NRows>::createFileSpace() const
{
	hsize_t     dims[2];
	dims[0] = 0;
	dims[1] = NRows;
	
	int ndim = 2;
	
	hsize_t maximumDims[2];
	maximumDims[0] = H5S_UNLIMITED;
	maximumDims[1] = NRows;
		
	return H5Screate_simple(ndim, dims, maximumDims);
}

template<int DataRank, int NRows>
hid_t H2dDataset<DataRank, NRows>::createMemSpace(hsize_t ncols) const
{
	hsize_t     dims[2];
	dims[0] = ncols;
	dims[1] = NRows;
	
	int ndim = 2;
	return H5Screate_simple(ndim, dims, NULL);
}

template<int DataRank, int NRows>
hid_t H2dDataset<DataRank, NRows>::createMemSpace(hdata::Select2DPart * part) const
{
	int ndim = 2;
	hid_t s = H5Screate_simple(ndim, part->count, NULL);
	return s;
}

template<int DataRank, int NRows>
hid_t H2dDataset<DataRank, NRows>::dataType()
{
	hid_t r = 0;
	switch (DataRank) {
		case hdata::TChar:
			r = H5T_NATIVE_CHAR;
			break;
		case hdata::TShort:
			r = H5T_NATIVE_SHORT;
			break;
		case hdata::TInt:
			r = H5T_NATIVE_INT;
			break;
		case hdata::TLongInt:
			r = H5T_NATIVE_LONG;
			break;
		case hdata::TFloat:
			r = H5T_NATIVE_FLOAT;
			break;
		case hdata::TDouble:
			r = H5T_NATIVE_DOUBLE;
			break;
		default:
			break;
	}
	return r;
}

template<int DataRank, int NRows>
int H2dDataset<DataRank, NRows>::NumBitsPerPnt()
{
	int nbits = 0;
	switch (DataRank) {
		case hdata::TChar:
			nbits = 1;
			break;
		case hdata::TShort:
			nbits = 2;
			break;
		case hdata::TInt:
			nbits = 4;
			break;
		case hdata::TLongInt:
			nbits = 8;
			break;
		case hdata::TFloat:
			nbits = 4;
			break;
		case hdata::TDouble:
			nbits = 8;
			break;
		default:
			break;
	}
	return nbits;
}

template<int DataRank, int NRows>
int H2dDataset<DataRank, NRows>::NumBitsPerCol()
{ return NumBitsPerPnt() * NRows; }

template<int DataRank, int NRows>
int H2dDataset<DataRank, NRows>::objectType() const
{ return H5G_DATASET; }

template<int DataRank, int NRows>
char H2dDataset<DataRank, NRows>::write(char *data, hdata::Select2DPart * part)
{
	hsize_t dims[2];
	dims[0] = part->start[0] + part->count[0];
	dims[1] = part->start[1] + part->count[1];
	H5Dset_extent(fObjectId, dims);
	
	hid_t memSpace = createMemSpace(part);
	
	herr_t status;
	
	hid_t fileSpace = H5Dget_space(fObjectId);
	
	status = H5Sselect_hyperslab(fileSpace, H5S_SELECT_SET, part->start, NULL, part->count, NULL);
	if(status < 0) std::cout<<"\nERROR: failed to select hyperslab";
	
	status = H5Dwrite(fObjectId, dataType(), memSpace, fileSpace, H5P_DEFAULT, data);
	H5Sclose(memSpace);
	H5Sclose(fileSpace);
	return status >= 0;
}

template<int DataRank, int NRows>
char H2dDataset<DataRank, NRows>::read(char *data, hdata::Select2DPart * part)
{
	hid_t memSpace = createMemSpace(part);
	herr_t status;
	
	hid_t fileSpace = H5Dget_space(fObjectId);
	
	status = H5Sselect_hyperslab(fileSpace, H5S_SELECT_SET, part->start, NULL, part->count, NULL);
	if(status < 0) std::cout<<"\nERROR: failed to select hyperslab\n";

	status = H5Dread(fObjectId, dataType(), memSpace, fileSpace, H5P_DEFAULT, data);
	H5Sclose(memSpace);
	H5Sclose(fileSpace);
	return status >= 0;
}

template<int DataRank, int NRows>
int H2dDataset<DataRank, NRows>::numColumns()
{
	hid_t fileSpace = H5Dget_space(fObjectId);
	
	hsize_t current_dims[2];
	hsize_t max_dims[2];
	H5Sget_simple_extent_dims(fileSpace, current_dims, max_dims);

	return current_dims[0];
}
