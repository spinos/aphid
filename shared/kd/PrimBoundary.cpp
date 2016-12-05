#include <kd/PrimBoundary.h>
#include <kd/SplitEvent.h>

namespace aphid {

PrimBoundary::PrimBoundary() : 
m_grid(NULL),
m_numPrims(0)
{}

PrimBoundary::~PrimBoundary() 
{ if(m_grid) delete m_grid; }
    
float PrimBoundary::visitCost() const
{ return .2f * m_numPrims; }
	
const int & PrimBoundary::numPrims() const 
{ return m_numPrims; }

bool PrimBoundary::isEmpty() const
{ return m_numPrims < 1; }

void PrimBoundary::addPrimitive(const unsigned & idx)
{ 
	m_indices.insert(idx); 
	m_numPrims++;
}

void PrimBoundary::addPrimitiveBox(const BoundingBox & b)
{ m_primitiveBoxes.insert(b); }

void PrimBoundary::compressPrimitives()
{
	m_grid = new sdb::GridClustering();
	m_grid->setGridSize(getBBox().getLongestDistance() / 64.f);
	
	int i = 0;
	for(;i<m_numPrims; i++)
		m_grid->insertToGroup(*m_primitiveBoxes[i], i);
	
	std::cout<<"\n ctx grid n cell "<<m_grid->size();
}

const sdb::VectorArray<unsigned> & PrimBoundary::indices() const
{ return m_indices; }

sdb::GridClustering * PrimBoundary::grid()
{ return m_grid; }

const sdb::VectorArray<BoundingBox> & PrimBoundary::primitiveBoxes() const
{ return m_primitiveBoxes; }

bool PrimBoundary::isCompressed() const
{ return m_grid != NULL; }

void PrimBoundary::clearPrimitive()
{
	m_indices.clear();
	m_numPrims = 0;
}

void PrimBoundary::uncompressGrid(const sdb::VectorArray<BoundingBox> & boxSrc)
{
	clearPrimitive();
		
	m_grid->extractInside(m_indices, boxSrc, getBBox() );
		
	m_numPrims = m_indices.size();
		
	delete m_grid;
	m_grid = NULL;
}

bool PrimBoundary::decompress(const sdb::VectorArray<BoundingBox> & boxSrc, 
		bool forced)
{
	if(!grid()) return false;
	if(numPrims() < 1024 
		|| grid()->size() < 8
		|| forced) {

		uncompressGrid(boxSrc);
		
		return true;
	}
	return false;
}

void PrimBoundary::createGrid(const float & x)
{
	m_grid = new sdb::GridClustering();
	m_grid->setGridSize(x);
	m_grid->setDataExternal();
}

void PrimBoundary::addCell(const sdb::Coord3 & x, sdb::GroupCell * c)
{ m_grid->insertChildValue(x, c); }

void PrimBoundary::countPrimsInGrid()
{
	m_numPrims = 0;
	if(!m_grid) return;
	m_numPrims = m_grid->numElements();
}

bool PrimBoundary::canEndSubdivide(const SplitEvent * split) const
{
	if(split->isEmpty() ) return true;
	if(isCompressed() ) return false;
    return (split->getCost() > visitCost() );
}

void PrimBoundary::verbose() const
{
	std::cout<<"\n split source "
			<<getBBox()
			<<" n prims "<<numPrims()
			<<" visit cost "<<visitCost()
			<<"\n";
}

}
