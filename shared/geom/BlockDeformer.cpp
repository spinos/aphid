/*
 *  BlockDeformer.cpp
 *  
 *  deform a chain of blocks by bend x twist y roll z
 *
 *  Created by jian zhang on 8/10/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "BlockDeformer.h"
#include "geom/ATriangleMesh.h"
#include <math/Matrix44F.h>
#include <geom/ConvexShape.h>

namespace aphid {

namespace deform {

Block::Block(Block* parent) :
m_parentBlock(parent)
{
    if(parent)
        parent->addChild(this); 
}

Block::~Block()
{
	std::vector<Block* >::iterator it = m_childBlocks.begin();
	for(;it!=m_childBlocks.end();++it) {
		delete *it;
	}
	m_childBlocks.clear();
}

Block* Block::child(int i) const
{ return m_childBlocks[i]; }

void Block::addChild(Block* child)
{ m_childBlocks.push_back(child); }

int Block::numChildBlocks() const
{ return m_childBlocks.size(); }

Matrix44F* Block::tmR()
{ return &m_tm; }

const Matrix44F& Block::worldTm() const
{ return m_wtm; }

void Block::updateWorldTm()
{
	if(m_parentBlock)
		m_wtm = m_tm * m_parentBlock->worldTm();
	else 
		m_wtm = m_tm;
		
	std::vector<Block* >::iterator it = m_childBlocks.begin();
	for(;it!=m_childBlocks.end();++it) {
		(*it)->updateWorldTm();
	}
}

}

BlockDeformerBuilder::BlockDeformerBuilder()
{}

BlockDeformerBuilder::~BlockDeformerBuilder()
{
	m_blocks.clear();
}

void BlockDeformerBuilder::bindVertexToBlock(Vector3F& pnt, int& iblock) const
{
/// by y
    int yseg = (pnt.y - 0.1f) / m_ySegment;
    if(yseg > numBlocks() - 1)
        yseg = numBlocks() - 1;
/// to local
    pnt.y -= m_ySegment * (float)yseg;
    iblock = yseg;
}

void BlockDeformerBuilder::setYSeg(float x)
{ m_ySegment = x; }

void BlockDeformerBuilder::addBlock(deform::Block* b)
{ m_blocks.push_back(b); }

int BlockDeformerBuilder::numBlocks() const
{ return m_blocks.size(); }

deform::Block* BlockDeformerBuilder::getBlock(int i) const
{ return m_blocks[i]; }

BlockDeformer::BlockDeformer() :
m_numBlocks(0)
{ 
    memset(m_angles, 0, 12);
}

BlockDeformer::~BlockDeformer()
{
	if(m_numBlocks>0)
		delete m_blocks[0];
}

void BlockDeformer::createBlockDeformer(const ATriangleMesh* mesh,
				const BlockDeformerBuilder& builder)
{
	if(m_numBlocks>0)
		delete m_blocks[0];
/// copy blocks
	const int nb = builder.numBlocks();
	m_blocks = new BlockPtrType[nb];
	for(int i=0;i<nb;++i) {
		m_blocks[i] = builder.getBlock(i);
	}
	m_numBlocks = nb;
	
	const int nv = mesh->numPoints();
	m_bind.reset(new char[nv<<4]);
	for(int i=0;i<nv;++i) {
		Vector3F pv = mesh->points()[i];
		int iblock = 0;
		builder.bindVertexToBlock(pv, iblock);
		
		char* dst = &m_bind[i<<4];		
		memcpy(dst, &pv, 12);
		memcpy(&dst[12], &iblock, 4);
	}
}

void BlockDeformer::setBend(const float& x)
{ m_angles[0] = x; }

void BlockDeformer::setTwist(const float& x)
{ m_angles[1] = x; }

void BlockDeformer::setRoll(const float& x)
{ m_angles[2] = x; }

void BlockDeformer::setScaling(const float* x)
{ 
    m_angles[3] = x[0];
    m_angles[4] = x[1];
}

void BlockDeformer::deform(const ATriangleMesh * mesh)
{
    if(!mesh)
        return;
    
    setOriginalMesh(mesh);
	
	updateBlocks();
    
	Vector3F plocal;
	int iblock;
	const int & nv = mesh->numPoints();
	for(int i=0;i<nv;++i) {
		Vector3F& pv = points()[i];
		pv = mesh->points()[i];
		getBind(plocal, iblock, i);
		BlockPtrType bi = getBlock(iblock);
		plocal.x *= m_angles[4];
		plocal.y *= m_angles[3];
		plocal.z *= m_angles[4];
		pv = bi->worldTm().transform(plocal);
	}
	
	calculateNormal(mesh);
	
}

void BlockDeformer::updateBlocks()
{
    const float scaling = 1.f / (float)(m_numBlocks - 1);
    const float droll = rollAngle() * scaling;
    const float dtwist = twistAngle() * scaling;
    const float dbend = bendAngle() * scaling;
    Matrix33F zrm;
    zrm.rotateZ(droll);
    Matrix33F yrm;
    yrm.rotateY(dtwist);
    Matrix33F xrm;
    xrm.rotateX(dbend);
    
	for(int i=1;i<m_numBlocks;++i) {
		BlockPtrType bi = getBlock(i);
/// calculate local tm
        bi->tmR()->setRotation(xrm * yrm * zrm);
/// shrink to about 0.932 after 7 blocks
        bi->tmR()->scaleBy(.99f);
	}
	BlockPtrType br = getBlock(0);
	br->updateWorldTm();
}

const float& BlockDeformer::bendAngle() const
{ return m_angles[0]; }

const float& BlockDeformer::twistAngle() const
{ return m_angles[1]; }

const float& BlockDeformer::rollAngle() const
{ return m_angles[2]; }

BlockDeformer::BlockPtrType BlockDeformer::getBlock(int i)
{ return m_blocks[i]; }

const BlockDeformer::BlockPtrType BlockDeformer::getBlock(int i) const
{ return m_blocks[i]; }

void BlockDeformer::getBind(Vector3F& plocal, int& iblock, const int& i) const
{
	char* src = &m_bind[i<<4];		
	memcpy(&plocal, src, 12);
	memcpy(&iblock, &src[12], 4);
}

void BlockDeformer::getBlockTms(float* y) const
{
	for(int i=0;i<m_numBlocks;++i) {
		const BlockPtrType bi = getBlock(i);
		const Matrix44F& wtm = bi->worldTm();
		wtm.glMatrix(&y[i<<4]);
	}
}

}
