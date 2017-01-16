/*
 *  Sculptor.cpp
 *  btree
 *
 *  Created by jian zhang on 5/10/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "Sculptor.h"

namespace aphid {
namespace sdb {

Sculptor::Sculptor() 
{
	m_tree = new WorldGrid<Array<int, VertexP>, VertexP >;
	m_active = new ActiveGroup;
	m_active->reset();
	m_strength = 0.5f;
	m_topo = NULL;
	m_activeStageId = 0;
	appendStage();
	m_refP = new Array<int, Vector3F >;
}

Sculptor::~Sculptor()
{
	delete m_tree;
	delete m_active;
	std::deque<Array<int, VertexP> * >::iterator it = m_stages.begin();
	for(;it!=m_stages.end();++it) {
		(*it)->clear();
		delete *it;
	}
	m_stages.clear();
	delete m_refP;
}

void Sculptor::beginAddVertices(const float & gridSize)
{
	m_tree->clear();
	m_tree->setGridSize(gridSize);
	m_refP->clear();
}

void Sculptor::insertVertex(VertexP * v)
{ m_tree->insert((const float *)v->index->t1, v); }

void Sculptor::insertRefP(VertexP * v)
{ 
	Vector3F * p = new Vector3F;
	*p = *v->index->t1;
	m_refP->insert(v->key, p);
}

void Sculptor::endAddVertices()
{
    std::cout<<"\n grid count "<<m_tree->size();
    
	m_tree->calculateBBox();
	m_march.initialize(m_tree->boundingBox(), m_tree->gridSize());
}

void Sculptor::setSelectRadius(const float & x)
{
	m_active->threshold = x;
}

const float & Sculptor::selectRadius() const
{ return m_active->threshold; }

void Sculptor::setStrength(const float & x) 
{ m_strength = x; }

void Sculptor::setTopology(SimpleTopology * topo) 
{ m_topo = topo; }

void Sculptor::selectPoints(const Ray * incident)
{
	m_active->reset();
	m_active->incidentRay = *incident;
	
	Sequence<Coord3> added;
	BoundingBox touchedBox;
	if(!m_march.begin(*incident)) return;
	while(!m_march.end()) {
		const std::deque<Vector3F> coords = m_march.touched(selectRadius(), touchedBox);

		std::deque<Vector3F>::const_iterator it = coords.begin();
		for(; it != coords.end(); ++it) {
			const Coord3 c = m_tree->gridCoord((const float *)&(*it));
/// already tested
			if(added.find(c)) continue;
            
			added.insert(c);

			Pair<Entity *, Entity> sr = m_tree->findEntity(c);
			if(sr.index) {
				Array<int, VertexP> * pl = static_cast<Array<int, VertexP> * >(sr.index);
				intersect(pl, *incident);
			}
		}

		float tmin, tmax;
		touchedBox.intersect(*incident, &tmin, &tmax);
		/// std::cout<<" tmin "<<tmin<<" "<<m_active->meanDepth();
		if((tmin - m_active->minDepth() ) > selectRadius() ) {
		    break;
		}
		
		m_march.step();
	}
    
	m_active->finish();
}

void Sculptor::deselectPoints() 
{ 
#ifdef DEBUG
	std::cout<<"grid count "<<m_tree->size();
	std::cout<<" sel depth "<<m_active->depthMax - m_active->depthMin;
	std::cout<<" sel count "<<m_active->numActivePoints();
#endif
	finishStage();
	m_tree->calculateBBox();
	m_march.initialize(m_tree->boundingBox(), m_tree->gridSize());
}

bool Sculptor::intersect(Array<int, VertexP> * d, const Ray & ray)
{
	if(!d) return false;
    const int ndst = m_active->vertices->size();
	Vector3F pop;
	float tt;
	d->begin();
	while(!d->end() ) {
		Vector3F & p = *(d->value()->index->t1);
		pop = ray.closestPointOnRay(p, &tt);
/// select here
		if(p.distanceTo(pop) < selectRadius()) {
			VertexP * vert = d->value();
			
			float d = p.distanceTo(ray.m_origin);
			m_active->updateMinDepth(d);
			
			if(d - m_active->minDepth() < 2.f * selectRadius()) {
				addToStage(vert);
				addToActive(vert);
			}
		}
		d->next();
	}
	
	return m_active->vertices->size() > ndst;
}

void Sculptor::addToActive(VertexP * v)
{
	if(m_active->vertices->find(v->key) ) return;
	VertexP * av = new VertexP;
	av->key = v->key;
	av->index = v->index;
	m_active->vertices->insert(av->key, av);
}

void Sculptor::addToStage(VertexP * v)
{
	if(currentStage()->find(v->key) ) return;
	
	VertexP * av = new VertexP;
	av->key = v->key;
	av->index = new PNPrefW;
	av->index->t1 = v->index->t1;
	
/// Pref <- P
	*(av->index->t3) = *(v->index->t1);
	currentStage()->insert(av->key, av);
}

void Sculptor::appendStage()
{ m_stages.push_back(new sdb::Array<int, VertexP>() ); }

sdb::Array<int, VertexP> * Sculptor::currentStage()
{ return m_stages[m_activeStageId]; }

void Sculptor::finishStage() 
{
	if(!currentStage()->isEmpty() ) {
		sdb::Array<int, VertexP> * stg = m_stages[m_activeStageId];
		stg->begin();
		while(!stg->end()) {
			VertexP * vert = stg->value();
/// N <- P
			*(vert->index->t2) = *(vert->index->t1);
			stg->next();	
		}
		m_activeStageId++;
		if(m_activeStageId >= m_stages.size())
			appendStage();
		else 
			m_stages[m_activeStageId]->clear();
			
		if(m_stages.size() > 200) {
/// limit num stages
			delete m_stages[0];
			m_stages.erase(m_stages.begin() );
			m_activeStageId--;
		}
		
		std::cout<<"\n finish stage "<<m_activeStageId-1
			<<" add "<<m_activeStageId;
	}
}

void Sculptor::revertStage(sdb::Array<int, VertexP> * stage, bool isBackward)
{
	if(!stage) return;
	
	stage->begin();
	while(!stage->end()) {
		VertexP * vert = stage->value();
		const Vector3F p0 = *(vert->index->t1);
		if(isBackward) {
/// P <- Pref
			*(vert->index->t1) = *(vert->index->t3);
		}
		else
/// P <- N
			*(vert->index->t1) = *(vert->index->t2);
		m_tree->displace(vert, *(vert->index->t1), p0);
		stage->next();	
	}
	m_tree->calculateBBox();
}

void Sculptor::undo()
{
	if(m_activeStageId<1) return;
	sdb::Array<int, VertexP> * stg = m_stages[m_activeStageId-1];
	revertStage(stg);
    m_lastStage = stg;
	m_activeStageId--;
	if(m_activeStageId<0) m_activeStageId=0;
	
	std::cout<<"\n undo stage "<<m_activeStageId<<" current "<<m_activeStageId;
}

void Sculptor::redo()
{
	if(currentStage()->isEmpty() ) return;
	sdb::Array<int, VertexP> * stg = m_stages[m_activeStageId];
	revertStage(stg, false);
    m_lastStage = stg;
	m_activeStageId++;
	if(m_activeStageId>=m_stages.size() )
		appendStage();
		
	std::cout<<"\n redo stage "<<m_activeStageId-1<<" current "<<m_activeStageId;
}

WorldGrid<Array<int, VertexP>, VertexP > * Sculptor::allPoints() const 
{ return m_tree; }

ActiveGroup * Sculptor::activePoints() const 
{ return m_active; }

void Sculptor::pullPoints()
{ movePointsAlong(m_active->meanNormal, 0.04f * selectRadius()); }

void Sculptor::pushPoints()
{ movePointsAlong(m_active->meanNormal, -0.04f * selectRadius()); }

void Sculptor::pinchPoints()
{ movePointsToward(m_active->meanPosition, 0.04f); }

void Sculptor::spreadPoints()
{ movePointsToward(m_active->meanPosition, -0.04f); }

/// mean normal + from center
void Sculptor::inflatePoints()
{ 
	Vector3F nor = m_active->meanNormal;
	Array<int, VertexP> * vs = m_active->vertices;
	
	float wei, round;
	vs->begin();
	while(!vs->end()) {
		
		VertexP * l = vs->value();
		wei = *l->index->t4;
		
		const Vector3F p0(*(l->index->t1));
		
		Vector3F pn = *l->index->t2;
/// blow outwards
		if(pn.dot(nor) < 0.f) pn.reverse();
		
		round = cos(p0.distanceTo(m_active->meanPosition) / selectRadius() * 1.5f );
		pn *= round;
		pn += nor * round;
		
		*(l->index->t1) += pn * wei * m_strength * 0.1f;
	
		m_tree->displace(l, *(l->index->t1), p0);

		vs->next();
	}
	smoothPoints(0.4f);
}

void Sculptor::smudgePoints(const Vector3F & x)
{ movePointsAlong(x, 0.5f); }

void Sculptor::smoothPoints(float frac)
{	
	if(m_active->numSelected() < 1) return;
	Array<int, VertexP> * vs = m_active->vertices;
	Vector3F d;

	float wei;
	vs->begin();
	while(!vs->end()) {
		
		VertexP * l = vs->value();
		wei = *l->index->t4;
		
		const Vector3F p0(*(l->index->t1));
		
		m_topo->getDifferentialCoord(l->key, d);
		
		*(l->index->t1) -= d * frac * wei * m_strength;
	
		m_tree->displace(l, *(l->index->t1), p0);

		vs->next();
	}
}

void Sculptor::movePointsAlong(const Vector3F & d, const float & fac)
{
	if(m_active->numSelected() < 1) return;
	Array<int, VertexP> * vs = m_active->vertices;
	//std::cout<<"\n b4 move sel "<<vs->size();
	float wei;
	vs->begin();
	while(!vs->end()) {
		
		VertexP * l = vs->value();
		
		wei = *l->index->t4;
		
		const Vector3F p0(*(l->index->t1));
		
		*(l->index->t1) += d * fac * wei * m_strength;
	
		m_tree->displace(l, *(l->index->t1), p0);
		vs->next();
	}
}

void Sculptor::movePointsToward(const Vector3F & d, const float & fac, bool normalize, Vector3F * vmod)
{
	if(m_active->numSelected() < 1) return;
	Array<int, VertexP> * vs = m_active->vertices;
	
	Vector3F tod;
	float wei;
	vs->begin();
	while(!vs->end()) {
		
		VertexP * l = vs->value();
		wei = *l->index->t4;
		
		const Vector3F p0(*(l->index->t1));
		
		tod = d - *(l->index->t1);
		if(normalize) tod.normalize();
		*(l->index->t1) += tod * fac * wei * m_strength;
		if(vmod) {
			*(l->index->t1) += *vmod * wei * m_strength;
		}
	
		m_tree->displace(l, *(l->index->t1), p0);
		vs->next();
	}
}

Array<int, VertexP> * Sculptor::lastStage()
{ return m_lastStage; }

void Sculptor::clearCurrentStage()
{ m_stages[m_activeStageId]->clear(); }

void Sculptor::clearAllStages()
{
	int i=0;
	for(;i<m_stages.size();++i) m_stages[i]->clear();
	m_activeStageId = 0;
}

void Sculptor::erasePoints()
{
	Array<int, VertexP> * vs = m_active->vertices;
	
	Vector3F tod;
	float wei;
	vs->begin();
	while(!vs->end()) {
		
		VertexP * l = vs->value();
		wei = *l->index->t4;
		
		Vector3F * pr = m_refP->find(vs->key() );
		if(pr) {
			const Vector3F p0(*(l->index->t1));
		
			tod = *pr - *(l->index->t1);
			*(l->index->t1) += tod * wei * m_strength * .5f;
			
			m_tree->displace(l, *(l->index->t1), p0);
		}
		else {
			std::cout<<"\n sculpter error cannot find pref "<<vs->key();
		}
		
		vs->next();
	}
}

} // end namespace sdb
}