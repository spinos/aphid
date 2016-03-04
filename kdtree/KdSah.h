#pragma once
#include <SplitEvent.h>
#include <MinMaxBins.h>
#include <Boundary.h>
#include <boost/thread.hpp>
#include <GridClustering.h>

namespace aphid {

template <typename T>
class SahSplit : public Boundary {
    
	sdb::VectorArray<T> * m_source;
	GridClustering * m_grid;
	sdb::VectorArray<BoundingBox> m_primitiveBoxes;
    int * m_indices;
    MinMaxBins * m_bins;
	SplitEvent * m_event;
    int m_bestEventIdx;
	int m_numPrims;
    
public:
    SahSplit(int n, sdb::VectorArray<T> * source);
    virtual ~SahSplit();
	
	void initIndicesAndBoxes();
	void compressPrimitives();
    
	void setIndexAt(int idx, int val);
	int indexAt(int idx) const;
    
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
	void initEventsAlong(const BoundingBox & b, const int &axis);
	void updateEventsAlong(const BoundingBox & b, const int &axis);
    int splitAtLowestCost();
    bool byCutoffEmptySpace(int & dst);
    SplitEvent * splitAt(int axis, int idx) const; 
};

template <typename T>
SahSplit<T> * SahSplit<T>::GlobalSplitContext = NULL;

template <typename T>
SahSplit<T>::SahSplit(int n, sdb::VectorArray<T> * source) : m_indices(NULL),
m_grid(NULL)
{
	m_source = source;
    m_bins = new MinMaxBins[SplitEvent::Dimension];
	m_event = new SplitEvent[SplitEvent::NumEventPerDimension * SplitEvent::Dimension];
	if(n>0) m_indices = new int[n];
	m_numPrims = n;
}

template <typename T>
SahSplit<T>::~SahSplit()
{
    delete[] m_bins;
    delete[] m_event;
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
int SahSplit<T>::indexAt(int idx) const
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
	m_bins[axis].create(SplitEvent::NumBinPerDimension, b.getMin(axis), b.getMax(axis));
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
void SahSplit<T>::initEventsAlong(const BoundingBox & b, const int &axis)
{
	const float min = b.getMin(axis);
	const float delta = b.distance(axis) / SplitEvent::NumBinPerDimension;
	const int eventOffset = axis * SplitEvent::NumEventPerDimension;
	unsigned leftNumPrim, rightNumPrim;
	int i;
	for(i = 0; i < SplitEvent::NumEventPerDimension; i++) {
		SplitEvent & event = m_event[eventOffset + i];
		event.setBAP(b, axis, min + delta * (i + 1) );
		m_bins[axis].get(i, leftNumPrim, rightNumPrim);
		event.setLeftRightNumPrim(leftNumPrim, rightNumPrim);
	}
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
	
	const float thre = bb.getLongestDistance() * .1f;
	int axis;
	
	for(axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(bb.distance(axis) < thre) 
		    m_bins[axis].setFlat();		
	}
	
    calculateBins(bb);
	
	for(axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(m_bins[axis].isFlat())
			continue;
	
		initEventsAlong(bb, axis);
	}
	
	calculateSplitEvents(bb);
	
	m_bestEventIdx = splitAtLowestCost();
#if 1
	int lc = 0;
	if(byCutoffEmptySpace(lc)) {
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
int SahSplit<T>::splitAtLowestCost()
{
	float lowest = 10e28f;
	int result = 0;
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		for(int i = 1; i < SplitEvent::NumEventPerDimension - 1; i++) {
			const SplitEvent * e = splitAt(axis, i);
			if(e->getCost() < lowest && e->hasBothSides()) {
				lowest = e->getCost();
				result = i + SplitEvent::NumEventPerDimension * axis;
			}
		}
	}
    return result;
}

template <typename T>
bool SahSplit<T>::byCutoffEmptySpace(int & dst)
{
	const BoundingBox bb = getBBox();
	int res = -1;
	float vol, area, emptyVolume = -1.f;
	const int minHead = 2;
	const int maxTail = SplitEvent::NumEventPerDimension - 3;
	const int midSect = SplitEvent::NumEventPerDimension / 2;
	int i, head, tail;
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(m_bins[axis].isFlat() ) continue;
		
		area = bb.crossSectionArea(axis);
		
		head = 0;
		SplitEvent * cand = splitAt(axis, 0);
		if(cand->leftCount() == 0) {
			for(i = minHead; i < midSect; i++) {
				cand = splitAt(axis, i);
				if(cand->leftCount() == 0)
					head = i;
			}
			
			if(head > minHead) {
				vol = head * m_bins[axis].delta() * area;
				
				if(vol > emptyVolume) {
					emptyVolume = vol;
					res = SplitEvent::NumEventPerDimension * axis + head;
				}
			}
		}
		tail = SplitEvent::NumEventPerDimension - 1;
		cand = splitAt(axis, SplitEvent::NumEventPerDimension - 1);
		if(cand->rightCount() == 0) {
			for(i = maxTail; i > midSect; i--) {
				cand = splitAt(axis, i);
				if(cand->rightCount() == 0)
					tail = i;
			}
			
			if(tail < maxTail) {
				vol = (SplitEvent::NumEventPerDimension - tail) * m_bins[axis].delta() * area;
				
				if(vol > emptyVolume) {
					emptyVolume = vol;
					res = SplitEvent::NumEventPerDimension * axis + tail;
				}
			}
		}
	}
	if(res > 0) dst = res;
	
	return res>0;
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
SplitEvent * SahSplit<T>::splitAt(int axis, int idx) const
{ return &m_event[axis * SplitEvent::NumEventPerDimension + idx]; }

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