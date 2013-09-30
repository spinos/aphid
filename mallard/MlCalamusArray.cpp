/*
 *  MlCalamusArray.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/14/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "MlCalamusArray.h"

struct test {
    MlFeather * m_geo;
    unsigned m_faceIdx, m_bufStart;
	float m_patchU, m_patchV;
	float m_rotX, m_rotY;
	//float m_scale;
};

MlCalamusArray::MlCalamusArray()
{
	setIndex(0);
	setElementSize(sizeof(MlCalamus));
	
	std::cout<<"size of calamus "<<sizeof(MlCalamus)<<"\n";
	std::cout<<"size of MlFeather * "<<sizeof(MlFeather *)<<"\n";
	std::cout<<"size of float "<<sizeof(float)<<"\n";
	std::cout<<"size of unsigned "<<sizeof(unsigned)<<"\n";
	std::cout<<"size of test "<<sizeof(test)<<"\n";
	
	MlCalamus c;
	c.setPatchU(0.0001);
	c.setPatchV(0.024607);
	std::cout<<" u 0.0001 "<<c.patchU()<<"\n"<<" v 0.024607 "<<c.patchV()<<"\n";
}

MlCalamusArray::~MlCalamusArray()
{
	clear();
}
	
MlCalamus * MlCalamusArray::asCalamus(unsigned index)
{
	return (MlCalamus *)at(index);
}

MlCalamus * MlCalamusArray::asCalamus(unsigned index) const
{
	return (MlCalamus *)at(index);
}

MlCalamus * MlCalamusArray::asCalamus()
{
	return (MlCalamus *)current();
}

float MlCalamusArray::sortKeyAt(unsigned idx) const
{
	MlCalamus * c = asCalamus(idx);
	return (float)c->faceIdx();
}

void MlCalamusArray::swapElement(unsigned a, unsigned b)
{
	MlCalamus t = *asCalamus(a);
	*asCalamus(a) = *asCalamus(b);
	*asCalamus(b) = t;
}

