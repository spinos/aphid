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
    int * m_indices;
    int m_bestEventIdx;
	int m_numPrims;
    
public:
    SahSplit(int n, sdb::VectorArray<T> * source);
    virtual ~SahSplit();
	
	void initIndicesAndBoxes();
	void compressPrimitives();
    
	void setIndexAt(int idx, int val);
	const int & indexAt(int idx) const;
    
    SplitEvent * bestSplit();
    void partition(SahSplit * leftSplit, SahSplit * rightSplit);
	
	const int & numPrims() const 
	{ return m_numPrims; }
	
	float visitCost() const
	{ return 2.f * m_numPrims; }
	
	bool isEmpty() const
	{ return m_numPrims < 1; }
	
	sdb::VectorArray<T> * source()
	{ return m_source; }
	
	void verbose() const;
	
	static SahSplit * GlobalSplitContext;
	
protected:
	const sdb::VectorArray<BoundingBox> & primitiveBoxes() const;
	GridClustering * grid();
	
private:
    void calculateBins(const BoundingBox & b);
	void calculateSplitEvents(const BoundingBox & b);
	void binningAlong(const BoundingBox & b, int axis);
	void updateEventsAlong(const BoundingBox & b, const int &axis);
    
};

template <typename T>
SahSplit<T> * SahSplit<T>::GlobalSplitContext = NULL;

template <typename T>
SahSplit<T>::SahSplit(int n, sdb::VectorArray<T> * source) : m_indices(NULL),
m_grid(NULL)
{
	m_source = source;
    if(n>0) m_indices = new int[n];
	m_numPrims = n;
}

template <typename T>
SahSplit<T>::~SahSplit()
{
    if(m_indices) delete[] m_indices;
	if(m_grid) delete m_grid;
}

template <typename T>
void SahSplit<T>::initIndicesAndBoxes() 
{
	int i = 0;
	for(;i<m_numPrims; i++) {
		m_indices[i] = i;
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
void SahSplit<T>::setIndexAt(int idx, int val)
{ m_indices[idx] = val; }

template <typename T>
const int & SahSplit<T>::indexAt(int idx) const
{ return m_indices[idx]; }

template <typename T>
void SahSplit<T>::calculateBins(const BoundingBox & b)
{
	int axis;
#if 0
	boost::thread boxThread[3];
	
	for(axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(m_bins[axis].isFlat())
			continue;
		boxThread[axis] = boost::thread(boost::bind(&SahSplit::binningAlong, this, b, axis));
	}
	
	for(axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(m_bins[axis].isFlat())
			continue;
		boxThread[axis].join();
	}
#else
	for(axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(m_bins[axis].isFlat())
			continue;
		binningAlong(b, axis);
	}
#endif
}

template <typename T>
void SahSplit<T>::binningAlong(const BoundingBox & b, int axis)
{	
	const sdb::VectorArray<BoundingBox> & primBoxes = SahSplit<T>::GlobalSplitContext->primitiveBoxes();
	
	for(int i = 0; i < m_numPrims; i++) {
		const BoundingBox * primBox = primBoxes[indexAt(i)];
		m_bins[axis].add(primBox->getMin(axis), primBox->getMax(axis));
	}
	
	m_bins[axis].scan();
}

template <typename T>
void SahSplit<T>::calculateSplitEvents(const BoundingBox & b)
{    
#if 0
    boost::thread boxThread[3];
	
	int axis;	
	for(axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(m_bins[axis].isFlat())
			continue;
		boxThread[axis] = boost::thread(boost::bind(&SahSplit::updateEventsAlong, this, b, axis));
	}
	
	for(axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(m_bins[axis].isFlat())
			continue;
		boxThread[axis].join();
	}
#else
	int axis;	
	for(axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(m_bins[axis].isFlat())
			continue;
			
		updateEventsAlong(b, axis);
	}
#endif
}

template <typename T>
void SahSplit<T>::updateEventsAlong(const BoundingBox & b, const int &axis)
{
	SplitEvent * eventOffset = &m_event[axis * SplitEvent::NumEventPerDimension];
	
    const float min = b.getMin(axis);
	const float delta = b.distance(axis) / SplitEvent::NumBinPerDimension;
	int g, minGrid, maxGrid;
	const sdb::VectorArray<BoundingBox> & primBoxes = SahSplit<T>::GlobalSplitContext->primitiveBoxes();
	int i;	
    for(i = 0; i < m_numPrims; i++) {
		const int iprim = indexAt(i);
		const BoundingBox * primBox = primBoxes[iprim];

		minGrid = (primBox->getMin(axis) - min) / delta;
		if(minGrid < 0) minGrid = 0;
		
		for(g = minGrid; g < SplitEvent::NumEventPerDimension; g++)
			eventOffset[g].updateLeftBox(*primBox);

		maxGrid = (primBox->getMax(axis) - min) / delta;
		
		if(maxGrid > SplitEvent::NumEventPerDimension) maxGrid = SplitEvent::NumEventPerDimension;

		for(g = maxGrid; g > 0; g--)
			eventOffset[g - 1].updateRightBox(*primBox);
		
	}
	
	for(i = 0; i < SplitEvent::NumEventPerDimension; i++)
		eventOffset[i].calculateCost(b.area());
}

template <typename T>
SplitEvent * SahSplit<T>::bestSplit()
{
	const BoundingBox bb = getBBox();
	
	initBins(bb);
	calculateBins(bb);
	initEvents(bb);
	calculateSplitEvents(bb);
	
	m_bestEventIdx = splitAtLowestCost();
#if 1
	int lc = 0;
	if(byCutoffEmptySpace(lc, bb)) {
		if(m_event[lc].getCost() < m_event[m_bestEventIdx].getCost() * 2.f)
			m_bestEventIdx = lc;
#if 0
			std::cout<<" cutoff at "
				<<lc/SplitEvent::NumEventPerDimension
				<<":"
				<<lc%SplitEvent::NumEventPerDimension;
#endif
	}
#endif
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
		const int iprim = indexAt(i);
		T * geo = m_source->get(iprim);
		const BoundingBox & primBox = geo->bbox();
		
		side = e.side(primBox);
		if(side < 2) {
			if(primBox.touch(leftBox)) {
				leftSplit->setIndexAt(leftCount, iprim);
				leftCount++;
			}
		}
		if(side > 0) {
			if(primBox.touch(rightBox)) {
				rightSplit->setIndexAt(rightCount, iprim);
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