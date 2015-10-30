#pragma once
#include <SplitEvent.h>
#include <MinMaxBins.h>
#include <Boundary.h>
#include <boost/thread.hpp>

template <typename T>
class SahSplit : public Boundary {
    
    T ** m_input;
    MinMaxBins * m_bins;
	SplitEvent * m_event;
    int m_bestEventIdx;
	int m_numPrims;
    
public:
    SahSplit(int n);
    virtual ~SahSplit();
    
    void set(int idx, T * x);
    
    SplitEvent * bestSplit();
    void partition(SahSplit * leftSplit, SahSplit * rightSplit);
	
	int numPrims() const 
	{ return m_numPrims; }
	
	float visitCost() const
	{ return 1.99f * m_numPrims; }
	
	bool isEmpty() const
	{ return m_numPrims < 1; }
	
	void verbose() const;
	
protected:

private:
    void calculateBins(const BoundingBox & b);
	void calculateSplitEvents(const BoundingBox & b);
	void initEventsAlong(const BoundingBox & b, const int &axis);
	void updateEventsAlong(const BoundingBox & b, const int &axis);
    int splitAtLowestCost();
    bool byCutoffEmptySpace(int & dst);
    SplitEvent * splitAt(int axis, int idx) const; 
};

template <typename T>
SahSplit<T>::SahSplit(int n) : m_input(NULL)
{
    m_bins = new MinMaxBins[SplitEvent::Dimension];
	m_event = new SplitEvent[SplitEvent::NumEventPerDimension * SplitEvent::Dimension];
	if(n>0) m_input = new T *[n];
	m_numPrims = n;
}

template <typename T>
SahSplit<T>::~SahSplit()
{
    delete[] m_bins;
    delete[] m_event;
	if(m_input) delete[] m_input;
}

template <typename T>
void SahSplit<T>::set(int idx, T * x)
{ m_input[idx] = x; }

template <typename T>
void SahSplit<T>::calculateBins(const BoundingBox & b)
{
    for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		//printf("bbox size %f\n", m_bbox.getMax(axis) - m_bbox.getMin(axis));
		if(b.distance(axis) < 10e-4f) {
		    //printf("bbox[%i] is flat", axis);
			m_bins[axis].setFlat();
			continue;
		}
		m_bins[axis].create(SplitEvent::NumBinPerDimension, b.getMin(axis), b.getMax(axis));
	
		for(int i = 0; i < m_numPrims; i++) {
            const BoundingBox primBox = m_input[i]->bbox();
			m_bins[axis].add(primBox.getMin(axis), primBox.getMax(axis));
		}
		
		m_bins[axis].scan();
	}
}

template <typename T>
void SahSplit<T>::calculateSplitEvents(const BoundingBox & b)
{    
    boost::thread boxThread[3];
	
	int axis;
	for(axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(m_bins[axis].isFlat())
			continue;
		boxThread[axis] = boost::thread(boost::bind(&SahSplit::initEventsAlong, this, b, axis));
	}
	
	for(axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(m_bins[axis].isFlat())
			continue;
		boxThread[axis].join();
	}
	
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
    const float min = b.getMin(axis);
	const float delta = b.distance(axis) / SplitEvent::NumBinPerDimension;
	const int eventOffset = axis * SplitEvent::NumEventPerDimension;
	BoundingBox tightBox;
	int i;	
    for(i = 0; i < m_numPrims; i++) {
		const BoundingBox & primBox = m_input[i]->bbox();

		int minGrid = (primBox.getMin(axis) - min) / delta;
		if(minGrid < 0) minGrid = 0;
		
		for(int g = minGrid; g < SplitEvent::NumEventPerDimension; g++) {
			m_input[i]->intersect(m_event[eventOffset + g].leftBound(), &tightBox );
			m_event[eventOffset + g].updateLeftBox(tightBox);
		}

		int maxGrid = (primBox.getMax(axis) - min) / delta;
		if(maxGrid > SplitEvent::NumEventPerDimension) maxGrid = SplitEvent::NumEventPerDimension;

		for(int g = maxGrid; g > 0; g--) {
			m_input[i]->intersect(m_event[eventOffset + g - 1].rightBound(), &tightBox );
			m_event[eventOffset + g - 1].updateRightBox(tightBox);
		}
	}
	
	for(i = 0; i < SplitEvent::NumEventPerDimension; i++)
		m_event[eventOffset + i].calculateCost(b.area());
}

template <typename T>
SplitEvent * SahSplit<T>::bestSplit()
{
	const BoundingBox bb = getBBox();
    calculateBins(bb);
	calculateSplitEvents(bb);
	
	m_bestEventIdx = splitAtLowestCost();
#if 1
	int lc = 0;
	if(byCutoffEmptySpace(lc)) {
		if(m_event[lc].getCost() < m_event[m_bestEventIdx].getCost() * 2.f)
			m_bestEventIdx = lc;
			std::cout<<" cutoff at "
				<<lc/SplitEvent::NumEventPerDimension
				<<":"
				<<lc%SplitEvent::NumEventPerDimension;
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
	int i, head, tail;
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(axis == 0) area = bb.distance(1) * bb.distance(2);
		else if(axis == 1) area = bb.distance(2) * bb.distance(0);
		else if(axis == 2) area = bb.distance(0) * bb.distance(1);
		
		head = 4;
		SplitEvent * cand = splitAt(axis, 4);
		if(cand->leftCount() == 0) {
			for(i = 4; i < SplitEvent::NumEventPerDimension - 1; i++) {
				cand = splitAt(axis, i);
				if(cand->leftCount() == 0)
					head = i;
			}
			
			if(head > 4) {
				vol = area * head;
				if(vol > emptyVolume) {
					emptyVolume = vol;
					res = SplitEvent::NumEventPerDimension * axis + head;
				}
			}
		}
		
		tail = SplitEvent::NumEventPerDimension - 5;
		cand = splitAt(axis, SplitEvent::NumEventPerDimension - 5);
		if(cand->rightCount() == 0) {
			for(i = SplitEvent::NumEventPerDimension - 5; i > 1; i--) {
				cand = splitAt(axis, i);
				if(cand->rightCount() == 0)
					tail = i;
			}
			
			if(tail < SplitEvent::NumEventPerDimension - 5) {
				vol = area * (SplitEvent::NumEventPerDimension - tail);
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
		const BoundingBox & primBox = m_input[i]->bbox();
		
		side = e.side(primBox);
		if(side < 2) {
			if(primBox.touch(leftBox)) {
				leftSplit->set(leftCount, m_input[i]);
				leftCount++;
			}
		}
		if(side > 0) {
			if(primBox.touch(rightBox)) {
				rightSplit->set(rightCount, m_input[i]);
				rightCount++;
			}
		}
		
	}
	std::cout<<"\n partition "
			<<leftCount
			<<"|"<<rightCount
			<<"\n";
}

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
//:~