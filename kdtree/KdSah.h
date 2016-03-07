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
	
	bool decompressGrid(bool forced = false);
	
	void verbose() const;
	
	static SahSplit * GlobalSplitContext;
	
protected:
	const sdb::VectorArray<BoundingBox> & primitiveBoxes() const;
	GridClustering * grid();
	void createGrid(const float & x);
	void addCell(const sdb::Coord3 & x, GroupCell * c);
	void countPrimsInGrid();
	void addPrimitive(const unsigned & idx);
	
private:
	void partitionCompress(const SplitEvent * e,
					const BoundingBox & leftBox, const BoundingBox & rightBox,
						SahSplit * leftCtx, SahSplit * rightCtx);

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
	m_grid->setGridSize(getBBox().getLongestDistance() / 48.f);
	
	int i = 0;
	for(;i<m_numPrims; i++)
		m_grid->insertToGroup(*m_primitiveBoxes[i], i);
	
	std::cout<<"\n ctx grid n cell "<<m_grid->size();
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
				m_indices,
				SahSplit<T>::GlobalSplitContext->primitiveBoxes(),
				bb );
	
	initEvents(bb);
	
	if(grid()) calculateCompressSplitEvents(grid(), bb);
	else calculateSplitEvents(bb,
				numPrims(),
				m_indices,
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
	// int leftCount = 0;
	// int rightCount = 0;
	int side;
	for(unsigned i = 0; i < m_numPrims; i++) {
		const unsigned iprim = *m_indices[i];
		const BoundingBox & primBox = *primBoxes[iprim];
		
		side = e->side(primBox);
		if(side < 2) {
			if(primBox.touch(leftBox)) {
				leftSplit->addPrimitive(iprim);
				// leftCount++;
			}
		}
		if(side > 0) {
			if(primBox.touch(rightBox)) {
				rightSplit->addPrimitive(iprim);
				// rightCount++;
			}
		}
	}

	// std::cout<<"\n partition "<<m_numPrims
	//		<<" -> "<<leftCount
	//		<<"|"<<rightCount;
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
				if(e->leftCount()<1) std::cout<<"\n\n warning left should be empty! \n\n";
				else leftCtx->addCell(grd->key(), grd->value() );
			}
			if(side > 0) {
				if(e->rightCount()<1) std::cout<<"\n\n warning right should be empty! \n\n";
				else rightCtx->addCell(grd->key(), grd->value() );
			}
		}
		grd->next();
	}
	
	if(e->leftCount() > 0) {
		leftCtx->countPrimsInGrid();
		leftCtx->decompressGrid();
	}
	if(e->rightCount() > 0) {
		rightCtx->countPrimsInGrid();
		rightCtx->decompressGrid();
	}
}

template <typename T>
const sdb::VectorArray<BoundingBox> & SahSplit<T>::primitiveBoxes() const
{ return m_primitiveBoxes; }

template <typename T>
GridClustering * SahSplit<T>::grid()
{ return m_grid; }

template <typename T>
void SahSplit<T>::createGrid(const float & x)
{
	m_grid = new GridClustering();
	m_grid->setGridSize(x);
	m_grid->setDataExternal();
}

template <typename T>
void SahSplit<T>::addCell(const sdb::Coord3 & x, GroupCell * c)
{ m_grid->insertChildValue(x, c); }

template <typename T>
void SahSplit<T>::countPrimsInGrid()
{
	m_numPrims = 0;
	if(!m_grid) return;
	m_numPrims = m_grid->numElements();
}

template <typename T>
bool SahSplit<T>::decompressGrid(bool forced)
{
	if(!m_grid) return false;
	if(m_numPrims < 1024 
		|| m_grid->size() < 32
		|| forced) {
/// reset
        m_indices.clear();
		m_numPrims = 0;
		
		const sdb::VectorArray<BoundingBox> & boxSrc = GlobalSplitContext->primitiveBoxes();
		m_grid->extractInside(m_indices, boxSrc, getBBox() );
		
		m_numPrims = m_indices.size();
		
		delete m_grid;
		m_grid = NULL;
		
		return true;
	}
	return false;
}

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