#pragma once
#include <kd/BaseBinSplit.h>
#include <kd/PrimBoundary.h>
#include <boost/thread.hpp>

namespace aphid {

template <typename T>
class SahSplit : public BaseBinSplit, public PrimBoundary {
    
	sdb::VectorArray<T> * m_source;
	
public:
    SahSplit(sdb::VectorArray<T> * source);
    virtual ~SahSplit();
	
	void initIndicesAndBoxes(const unsigned & num);
	
    SplitEvent * bestSplit();
    void partition(SahSplit * leftSplit, SahSplit * rightSplit);
	
	sdb::VectorArray<T> * source();
	bool decompressPrimitives(bool force=false);
	
	static SahSplit * GlobalSplitContext;
	
protected:
	
private:
	void partitionCompress(const SplitEvent * e,
					const BoundingBox & leftBox, const BoundingBox & rightBox,
						SahSplit * leftCtx, SahSplit * rightCtx);

};

template <typename T>
SahSplit<T> * SahSplit<T>::GlobalSplitContext = NULL;

template <typename T>
SahSplit<T>::SahSplit(sdb::VectorArray<T> * source) 
{ m_source = source; }

template <typename T>
SahSplit<T>::~SahSplit()
{}

template <typename T>
void SahSplit<T>::initIndicesAndBoxes(const unsigned & num) 
{
	clearPrimitive();
	
	for(unsigned i=0;i<num; i++) {
		addPrimitive(i);
		addPrimitiveBox(m_source->get(i)->calculateBBox() );
	}
	
	if(numPrims() > 1024) compressPrimitives();
}

template <typename T>
SplitEvent * SahSplit<T>::bestSplit()
{
	const BoundingBox & bb = getBBox();
	
	if(grid()) 
		//calcEvenBin(grid(), bb);
		calcSoftBin(grid(), bb);
	else 
		//calcEvenBin(numPrims(),
		calcSoftBin(numPrims(),
				indices(),
				SahSplit<T>::GlobalSplitContext->primitiveBoxes(),
				bb );
	
	initEvents(bb);
	
	if(grid()) 
	    calcEvent(grid(), bb);
	else 
	    calcEvent(bb,
				numPrims(),
				indices(),
				SahSplit<T>::GlobalSplitContext->primitiveBoxes() );
	
	calculateCosts(bb);
	splitAtLowestCost(bb);
	return split(m_bestEventIdx);
}

template <typename T>
void SahSplit<T>::partition(SahSplit * leftSplit, SahSplit * rightSplit)
{	
	const SplitEvent * e = split(m_bestEventIdx);
	
	BoundingBox leftBox = e->leftBound();
	BoundingBox rightBox = e->rightBound();

	leftSplit->setBBox(leftBox);
	rightSplit->setBBox(rightBox);

	if(grid()) 
		return partitionCompress(e, leftBox, rightBox, leftSplit, rightSplit);
		
	const sdb::VectorArray<BoundingBox> & primBoxes = GlobalSplitContext->primitiveBoxes();
	
	const unsigned n = numPrims();
	int side;
	for(unsigned i = 0; i < n; i++) {
		const unsigned iprim = *indices()[i];
		const BoundingBox & primBox = *primBoxes[iprim];
		
		side = e->side(primBox);
		if(side < 2) {
			if(primBox.touch(leftBox)) {
				leftSplit->addPrimitive(iprim);
			}
		}
		if(side > 0) {
			if(primBox.touch(rightBox)) {
				rightSplit->addPrimitive(iprim);
			}
		}
	}
}

template <typename T>
void SahSplit<T>::partitionCompress(const SplitEvent * e,
					const BoundingBox & leftBox, const BoundingBox & rightBox,
						SahSplit * leftCtx, SahSplit * rightCtx)
{
	GridClustering * grd = grid();
	const BoundingBox & bb = getBBox();
	
	if(e->leftCount() > 0)
		leftCtx->createGrid(grd->gridSize() );
	if(e->rightCount() > 0)
		rightCtx->createGrid(grd->gridSize() );
		
	int side;
	grd->begin();
	while (!grd->end() ) {
		const BoundingBox & primBox = grd->value()->m_box;
		if(primBox.touch(bb) ) {	
			side = e->side(primBox);
			if(side < 2) {
				if(e->leftCount()<1) {
					std::cout<<"\n warning left should be empty!"
					<<"\n "<<primBox;
					e->verbose();
				}
				else leftCtx->addCell(grd->key(), grd->value() );
			}
			if(side > 0) {
				if(e->rightCount()<1) {
					std::cout<<"\n warning right should be empty!"
					<<" \n"<<primBox;
					e->verbose();
				}
				else rightCtx->addCell(grd->key(), grd->value() );
			}
		}
		grd->next();
	}
	
	if(e->leftCount() > 0)
		leftCtx->decompressPrimitives();
		
	if(e->rightCount() > 0)
		rightCtx->decompressPrimitives();
		
}

template <typename T>
sdb::VectorArray<T> * SahSplit<T>::source()
{ return m_source; }
	
template <typename T>
bool SahSplit<T>::decompressPrimitives(bool force)
{ 
	if(!force) countPrimsInGrid();
	return decompress(GlobalSplitContext->primitiveBoxes(), force);
}

}
//:~