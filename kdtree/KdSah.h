#pragma once
#include <SplitEvent.h>
#include <MinMaxBins.h>
#include <boost/thread.hpp>

class TestBox : public BoundingBox
{
public:
    TestBox() {}
    virtual ~TestBox() {}
    BoundingBox calculateBBox() const
    { return * this; }
    BoundingBox bbox() const
    { return * this; }
};

template <int Num, typename T>
class SahSplit : public Boundary {
    
    T *m_input[Num];
    MinMaxBins * m_bins;
	SplitEvent * m_event;
    int m_bestEventIdx;
    
public:
    SahSplit();
    virtual ~SahSplit();
    
    void set(int idx, T * x);
    
    void subdivide(KdTreeNode * node);
    SplitEvent * bestSplit();
    
protected:

private:
    void calculateBins(const BoundingBox & b);
	void calculateSplitEvents(const BoundingBox & b);
	void updateEventBBoxAlong(const BoundingBox & b, const int &axis);
    int splitAtLowestCost();
    bool byCutoffEmptySpace(int & dst);
    SplitEvent * splitAt(int axis, int idx) const; 
};

template <int Num, typename T>
SahSplit<Num, T>::SahSplit()
{
    m_bins = new MinMaxBins[SplitEvent::Dimension];
	m_event = new SplitEvent[SplitEvent::NumEventPerDimension * SplitEvent::Dimension];
}

template <int Num, typename T>
SahSplit<Num, T>::~SahSplit()
{
    delete[] m_bins;
    delete[] m_event;
}

template <int Num, typename T>
void SahSplit<Num, T>::set(int idx, T * x)
{
    m_input[idx] = x;
    updateBBox(x->calculateBBox());
}

template <int Num, typename T>
void SahSplit<Num, T>::subdivide(KdTreeNode * node)
{
    const BoundingBox bb = getBBox();
    std::cout<<" "<<bb;
    calculateBins(bb);
	calculateSplitEvents(bb);
}

template <int Num, typename T>
void SahSplit<Num, T>::calculateBins(const BoundingBox & b)
{
    for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		//printf("bbox size %f\n", m_bbox.getMax(axis) - m_bbox.getMin(axis));
		if(b.distance(axis) < 10e-4) {
		    //printf("bbox[%i] is flat", axis);
			m_bins[axis].setFlat();
			continue;
		}
		m_bins[axis].create(SplitEvent::NumBinPerDimension, b.getMin(axis), b.getMax(axis));
	
		for(unsigned i = 0; i < Num; i++) {
            const BoundingBox primBox = m_input[i]->bbox();
			m_bins[axis].add(primBox.getMin(axis), primBox.getMax(axis));
		}
		
		m_bins[axis].scan();
	}
}

template <int Num, typename T>
void SahSplit<Num, T>::calculateSplitEvents(const BoundingBox & b)
{
    int dimOffset;
	unsigned leftNumPrim, rightNumPrim;
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(m_bins[axis].isFlat())
			continue;
        
		dimOffset = SplitEvent::NumEventPerDimension * axis;	
		const float min = b.getMin(axis);
		const float delta = b.distance(axis) / SplitEvent::NumBinPerDimension;
		for(int i = 0; i < SplitEvent::NumEventPerDimension; i++) {
			SplitEvent &event = m_event[dimOffset + i];
			event.setAxis(axis);
			event.setPos(min + delta * (i + 1));
			m_bins[axis].get(i, leftNumPrim, rightNumPrim);
			event.setLeftRightNumPrim(leftNumPrim, rightNumPrim);
		}
	}
    
    boost::thread boxThread[3];
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(m_bins[axis].isFlat())
			continue;
		boxThread[axis] = boost::thread(boost::bind(&SahSplit::updateEventBBoxAlong, this, b, axis));
	}
	
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(m_bins[axis].isFlat())
			continue;
		boxThread[axis].join();
	}
	
    const unsigned numEvent = SplitEvent::NumEventPerDimension * SplitEvent::Dimension;
	for(unsigned i = 0; i < numEvent; i++) {
		m_event[i].calculateCost(b.area());
	}
}

template <int Num, typename T>
void SahSplit<Num, T>::updateEventBBoxAlong(const BoundingBox & b, const int &axis)
{
    const float min = b.getMin(axis);
	const float delta = b.distance(axis) / SplitEvent::NumBinPerDimension;
	const int eventOffset = axis * SplitEvent::NumEventPerDimension;
/// todo accurate cut here	
    for(unsigned i = 0; i < Num; i++) {
		const BoundingBox &primBox = m_input[i]->bbox();

		int minGrid = (primBox.getMin(axis) - min) / delta;
		
		if(minGrid < 0) minGrid = 0;
		
		for(int g = minGrid; g < SplitEvent::NumEventPerDimension; g++)
			m_event[eventOffset + g].updateLeftBox(primBox);

		int maxGrid = (primBox.getMax(axis) - min) / delta;
		
		if(maxGrid > SplitEvent::NumEventPerDimension) maxGrid = SplitEvent::NumEventPerDimension;

		for(int g = maxGrid; g > 0; g--)
			m_event[eventOffset + g - 1].updateRightBox(primBox);
	}
}

template <int Num, typename T>
SplitEvent * SahSplit<Num, T>::bestSplit()
{
	m_bestEventIdx = splitAtLowestCost();
	
	int lc = 0;
	if(byCutoffEmptySpace(lc)) {
		if(m_event[lc].getCost() < m_event[m_bestEventIdx].getCost() * 2.f)
			m_bestEventIdx = lc;
	}
		
	return &m_event[m_bestEventIdx];
}

template <int Num, typename T>
int SahSplit<Num, T>::splitAtLowestCost()
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

template <int Num, typename T>
bool SahSplit<Num, T>::byCutoffEmptySpace(int & dst)
{
	int res = -1;
	float vol, emptyVolume = -1.f;
	int i, head, tail;
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		head = 0;
		SplitEvent * cand = splitAt(axis, 0);
		if(cand->leftCount() == 0) {
			for(i = 1; i < SplitEvent::NumEventPerDimension - 1; i++) {
				cand = splitAt(axis, i);
				if(cand->leftCount() == 0)
					head = i;
			}
			
			if(head > 2) {
				vol = head;
				if(vol > emptyVolume) {
					emptyVolume = vol;
					res = SplitEvent::NumEventPerDimension * axis + head;
				}
			}
		}
		tail = SplitEvent::NumEventPerDimension - 1;
		cand = splitAt(axis, SplitEvent::NumEventPerDimension - 1);
		if(cand->rightCount() == 0) {
			for(i = 1; i < SplitEvent::NumEventPerDimension - 1; i++) {
				cand = splitAt(axis, SplitEvent::NumEventPerDimension - 1 - i);
				if(cand->rightCount() == 0)
					tail = SplitEvent::NumEventPerDimension - 1 - i;
			}
			if(tail < SplitEvent::NumEventPerDimension - 3) {
				vol = SplitEvent::NumEventPerDimension - tail;
				if(vol > emptyVolume) {
					emptyVolume = vol;
					res = SplitEvent::NumEventPerDimension * axis + tail;
				}
			}
		}
	}
	if(res > 0) {
		dst = res;
		//printf("cutoff at %i: %i left %i right %i\n", res/SplitEvent::NumEventPerDimension,  res%SplitEvent::NumEventPerDimension, m_event[res].leftCount(), m_event[res].rightCount());
	}
	return res>0;
}

template <int Num, typename T>
SplitEvent * SahSplit<Num, T>::splitAt(int axis, int idx) const
{ return &m_event[axis * SplitEvent::NumEventPerDimension + idx]; }

