/*
 *  HDataset.cpp
 *  helloHdf
 *
 *  Created by jian zhang on 6/13/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "HDataset.h"
#include <iostream>

HDataset::HDataset(const std::string & path) : HObject(path)
{
}

char HDataset::validate()
{
	if(!exists())
		return 0;
		
	//if(!verifyDataSpace())
	//	return 0;
		
	return 1;
}

char HDataset::create(hid_t parentId)
{	
	hid_t fDataSpace = createFileDataSpace();
	
	if(fDataSpace < 0) std::cout<<"\nh data space create failed\n";
	
	hid_t m_createProps = H5Pcreate(H5P_DATASET_CREATE);
	if(m_createProps < 0) std::cout<<"\nh create property failed\n";
	
	int ndim = H5Sget_simple_extent_ndims(fDataSpace);
	hsize_t dims[3] = {0, 0, 0};
	hsize_t maxdims[3];
	hsize_t m_chunkSize[3] = {32, 0, 0};
	
	H5Sget_simple_extent_dims(fDataSpace, dims, maxdims);
	m_chunkSize[0] = dims[0] / 32;
	
	std::cout<<"d space n dim "<<ndim<<"\n";
	std::cout<<"d space "<<dims[0]<<" "<<dims[1]<<" "<<dims[2]<<" \n";
	std::cout<<"d chunk "<<m_chunkSize[0]<<" "<<m_chunkSize[1]<<" "<<m_chunkSize[2]<<" \n";
	
	if(H5Pset_chunk(m_createProps, ndim, m_chunkSize)<0) {
      printf("Error: fail to set chunk\n");
      return -1;
   }
   	
	fObjectId = H5Dcreate2(parentId, fObjectPath.c_str(), dataType(), fDataSpace, 
                          H5P_DEFAULT, m_createProps, H5P_DEFAULT);
		  
	if(fObjectId < 0) {
	    std::cout<<"\nh data set create failed\n";
		return 0;
	}
	H5Sclose(fDataSpace);
	H5Pclose(m_createProps);
	return 1;
}

char HDataset::open(hid_t parentId)
{
	fObjectId = H5Dopen(parentId, fObjectPath.c_str(), H5P_DEFAULT);
	
	if(fObjectId < 0)
		return 0;
		
	//fDataSpace = H5Dget_space(fObjectId);

	//if(fDataSpace<0)
	//	return 0;
		
	return 1;
}

void HDataset::close()
{
	H5Dclose(fObjectId);
}

int HDataset::objectType() const
{
	return H5G_DATASET;
}

hid_t HDataset::dataType()
{
	return H5T_NATIVE_FLOAT;
}

hid_t HDataset::createFileDataSpace() const
{
	hsize_t     dims[3];
	dims[0] = (fDimension[0] / 32 + 1) * 32;
	dims[1] = 0;
	dims[2] = 0;
	
	int ndim = 1;
	
	hsize_t maximumDims[3];
	maximumDims[0] = H5S_UNLIMITED;
	maximumDims[1] = 0;
	maximumDims[2] = 0;
		
	return H5Screate_simple(ndim, dims, maximumDims);
}

hid_t HDataset::createMemDataSpace() const
{
	hsize_t     dims[3];
	dims[0] = fDimension[0];
	dims[1] = 0;
	dims[2] = 0;
	
	int ndim = 1;
		
	return H5Screate_simple(ndim, dims, NULL);
}

/*
char HDataset::verifyDataSpace()
{	
	open();
		
	char res = dimensionMatched();
		
	close();
	return res;
}

char HDataset::dimensionMatched()
{
	int rank = H5Sget_simple_extent_ndims(fDataSpace);
	
	if(rank != dataSpaceNumDimensions()){
		FileIO.fCurrentError = HDocument::eLackOfDataSpaceDimension;
		return 0;
	}
	
	hsize_t     dims_out[3];
	H5Sget_simple_extent_dims(fDataSpace, dims_out, NULL);
	
	for(int i=0; i<rank; i++) {
		if(dims_out[i] != (fDimension[i] / 32 + 1) * 32) {
			FileIO.fCurrentError = HDocument::eDataSpaceDimensionMissMatch;
			return 0;
		}
	}
	return 1;
}
*/

char HDataset::hasEnoughSpace() const
{
	hid_t spaceId = H5Dget_space(fObjectId);
	hsize_t dims[3];
	hsize_t maxdims[3];
	H5Sget_simple_extent_dims(spaceId, dims, maxdims);
	int ndim = H5Sget_simple_extent_ndims(spaceId);
	for(int i = 0; i < ndim; i++) {
		if(dims[i] < fDimension[i]) {
			std::cout<<" data space dim["<<i<<"] = "<<dims[i]<<" not enough for "<<fDimension[i]<<"\n";
			return 0;
		}
		else {
			std::cout<<" data space dim["<<i<<"] = "<<dims[i]<<" is enough for "<<fDimension[i]<<"\n";
		}
	}
	return 1;
}

char HDataset::write()
{
	float * data = new float[fDimension[0]*fDimension[1]];
    hsize_t i, j;
   for(j = 0; j < fDimension[0]; j++)
	for(i = 0; i < fDimension[1]; i++)
	    data[j*fDimension[1] +i] = i + j;
		
	write((char *)data);
	
	delete[] data;
	return 1;
}

char HDataset::read()
{
	float *         data = new float[fDimension[0]*fDimension[1]];
	read((char *)data);
	
	hsize_t i, j;				
	for(j = 0; j < fDimension[0]; j++) {
	for(i = 0; i < fDimension[1]; i++) {
		std::cout<<" "<<data[j*fDimension[1]+i];
	}
		std::cout<<" \n";
	}
	
	delete[] data;
	return 1;
}

char HDataset::write(char *data)
{
	std::cout<<"to write "<<fObjectPath<<"\n";
	hid_t memSpace = createMemDataSpace();
	if(memSpace > 0) std::cout<<"dspace success\n";
	else std::cout<<"dspace failed\n";
	
	hid_t spaceId = H5Dget_space(fObjectId);
	
	herr_t status = H5Dwrite(fObjectId, dataType(), H5S_ALL, memSpace, H5P_DEFAULT, data);
	H5Sclose(memSpace);
	if(status < 0)
		return 0;
		
	std::cout<<"wrote "<<fObjectPath<<"\n";
	return 1;
}

char HDataset::read(char *data)
{
	hid_t fDataSpace = createMemDataSpace();
	herr_t status = H5Dread(fObjectId, dataType(), H5S_ALL, fDataSpace, H5P_DEFAULT, data);
	H5Sclose(fDataSpace);
	if(status < 0) {
		std::cout<<"float write failed";
		return 0;
	}	
	return 1;
}

int HDataset::dataSpaceNumDimensions() const
{
	return 1;
}

void HDataset::dataSpaceDimensions(int dim[3]) const
{
	hid_t spaceId = H5Dget_space(fObjectId);
	hsize_t dims[3];
	hsize_t maxdims[3];
	H5Sget_simple_extent_dims(spaceId, dims, maxdims);
	
	dim[0] = dims[0];
	dim[1] = dims[1];
	dim[2] = 0;
}

void HDataset::resize()
{
	std::cout<<"resize data to accommodate "<<fDimension[0]<<"\n";
	hsize_t size[1] = {(fDimension[0] / 32 + 1) * 32};
	std::cout<<"targeting "<<(fDimension[0] / 32 + 1) * 32<<"\n";
	herr_t status = H5Dset_extent(fObjectId, size);
	if(status < 0) std::cout<<"resize failed\n";
	
	int dims[3];
	dataSpaceDimensions(dims);
	
	if(dims[0] != size[0])
		std::cout<<"failed to resize to "<<size[0]<<"\n";
	else
		std::cout<<"success resized to"<<size[0]<<"\n";
}
