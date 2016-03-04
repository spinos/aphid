#pragma once
#include <BaseBinSplit.h>
#include <Boundary.h>
#include <boost/thread.hpp>
#include <GridClustering.h>

namespace aphid {

template <typename T>
class SahSplit : public BaseBinSplit, public Boundary {
    
	sdb::VectorArray<T> * m_source;
	GridClustering * m_grid;
	sdb::VectorArray<BoundingBox> m_primitiveBoxes;
    sdb::VectorArray<unsigned> m_indices;
    unsigned m_numPrims;
    
public:
    SahSplit(sdb::VectorArray<T> * source);
    virtual ~SahSplit();
	
	void initIndicesAndBoxes(const unsigned & num);
	void compressPrimitives();
    
    SplitEvent * bestSplit();
    void partition(SahSplit * leftSplit, SahSplit * rightSplit);
	
	const unsigned & numPrims() const 
	{ return m_numPrims; }
	
	float visitCost() const
	{ return 2.f * m_numPrims; }
	
	bool isEmpty() const
	{ return m_numPrims < 1; }
	
	sdb::VectorArray<T> * source()
	{ return m_source; }
	
	const unsigned & indexAt(const unsigned & idx) const;
	
	void verbose() const;
	
	static SahSplit * GlobalSplitContext;
	
protected:
	const sdb::VectorArray<BoundingBox> & primitiveBoxes() const;
	GridClustering * grid();
	void addPrimitive(const unsigned & idx);
	
private:

};

template <typename T>
SahSplit<T> * SahSplit<T>::GlobalSplitContext = NULL;

template <typename T>
SahSplit<T>::SahSplit(sdb::VectorArray<T> * source) : m_indices(NULL),
m_grid(NULL)
{
	m_source = source;
    m_numPrims = 0;
}

template <typename T>
SahSplit<T>::~SahSplit()
{
    if(m_grid) delete m_grid;
}

template <typename T>
void SahSplit<T>::initIndicesAndBoxes(const unsigned & num) 
{
	m_numPrims = num;
	unsigned i = 0;
	for(;i<m_numPrims; i++) {
		m_indices.insert(i);
		m_primitiveBoxes.insert(m_source->get(i)->calculateBBox() );
	}
}

template <typename T>
void SahSplit<T>::compressPrimitives()
{
	m_grid = new GridClustering();
	m_grid->setGridSize(getBBox().getLongestDistance() / 32.f);
	
	int i = 0;
	for(;i<m_numPrims; i++)
		m_grid->insertToGroup(*m_primitiveBoxes[i], i);
	
	std::cout<<"\n ctx grid n cell "<<m_grid->size();
}

template <typename T>
SplitEvent * SahSplit<T>::bestSplit()
{
	const BoundingBox bb = getBBox();
	
	initBins(bb);
	const unsigned n = numPrims();
	calculateBins(n,
				m_indices,
				SahSplit<T>::GlobalSplitContext->primitiveBoxes() );
	initEvents(bb);
	calculateSplitEvents(bb,
				n,
				m_indices,
				SahSplit<T>::GlobalSplitContext->primitiveBoxes() );
	
	calculateCosts(bb);
	splitAtLowestCost(bb);
	return &m_event[m_bestEventIdx];
}

template <typename T>
void SahSplit<T>::partition(SahSplit * leftSplit, SahSplit * rightSplit)
{	
	SplitEvent & e = m_event[m_bestEventIdx];

	BoundingBox leftBox = e.leftBound();
	BoundingBox rightBox = e.rightBound();

	leftSplit->setBBox(leftBox);
	rightSplit->setBBox(rightBox);

	int leftCount = 0;
	int rightCount = 0;
	int side;
	for(unsigned i = 0; i < m_numPrims; i++) {
		const unsigned iprim = *m_indices[i];
		T * geo = m_source->get(iprim);
		const BoundingBox & primBox = geo->bbox();
		
		side = e.side(primBox);
		if(side < 2) {
			if(primBox.touch(leftBox)) {
				leftSplit->addPrimitive(iprim);
				leftCount++;
			}
		}
		if(side > 0) {
			if(primBox.touch(rightBox)) {
				rightSplit->addPrimitive(iprim);
				rightCount++;
			}
		}
		
	}

	// std::cout<<"\n partition "<<m_numPrims
	//		<<" -> "<<leftCount
	//		<<"|"<<rightCount;
}

template <typename T>
const sdb::VectorArray<BoundingBox> & SahSplit<T>::primitiveBoxes() const
{ return m_primitiveBoxes; }

template <typename T>
GridClustering * SahSplit<T>::grid()
{ return m_grid; }

template <typename T>
void SahSplit<T>::addPrimitive(const unsigned & idx)
{ 
	m_indices.insert(idx); 
	m_numPrims++;
}

template <typename T>
const unsigned & SahSplit<T>::indexAt(const unsigned & idx) const
{ return *m_indices[idx]; }

template <typename T>
void SahSplit<T>::verbose() const
{
	std::cout<<"\n split source "
			<<getBBox()
			<<" n prims "<<numPrims()
			<<" visit cost "<<visitCost()
			<<"\n";
}

}
//:~