#include "DynamicWorldInterface.h"
#include "CudaDynamicWorld.h"
#include <CudaTetrahedronSystem.h>
#include <CudaBroadphase.h>
#include <CudaNarrowphase.h>
#include <SimpleContactSolver.h>
#include <BaseBuffer.h>
#include <CUDABuffer.h>
#include <GeoDrawer.h>
#include <stripedModel.h>
#include <CudaLinearBvh.h>
#include <radixsort_implement.h>
#include <simpleContactSolver_implement.h>
#include <tetrahedron_math.h>
#include <boost/format.hpp>

#define GRDW 57
#define GRDH 57
#define NTET 3600
#define NPNT 14400

struct A {
    // mat33 Ke[4][4];
    mat33 Re;
    //float3 B[4]; 
    //float3 e1, e2, e3;
    //float volume;
    //float plastic[6];
};

DynamicWorldInterface::DynamicWorldInterface() 
{
    m_faultyPair[0] = 0; m_faultyPair[1] = 1;
    std::cout<<" size of A "<<sizeof(A)<<"\n";
    m_pairCache = new BaseBuffer;
    m_tetPnt = new BaseBuffer;
	m_tetInd = new BaseBuffer;
	m_pointStarts = new BaseBuffer;
	m_indexStarts = new BaseBuffer;
	m_constraint = new BaseBuffer;
	m_contactPairs = new BaseBuffer;
	m_contact = new BaseBuffer;
	m_pairsHash = new BaseBuffer;
	m_linearVelocity = new BaseBuffer;
	m_angularVelocity = new BaseBuffer;
	m_mass = new BaseBuffer;
	m_split = new BaseBuffer;
	m_deltaJ = new BaseBuffer;
}

DynamicWorldInterface::~DynamicWorldInterface() {}

void DynamicWorldInterface::create(CudaDynamicWorld * world)
{
    CudaTetrahedronSystem * tetra = new CudaTetrahedronSystem;
	tetra->create(NTET, NPNT);
	float * hv = &tetra->hostV()[0];
	
	unsigned i, j;
	float vy = 3.95f;
	float vrx, vry, vrz, vr, vs;
	for(j=0; j < GRDH; j++) {
		for(i=0; i<GRDW; i++) {
		    vs = 1.75f + RandomF01() * 1.5f;
			Vector3F base(9.3f * i, 9.3f * j, 0.f * j);
			Vector3F right = base + Vector3F(1.75f, 0.f, 0.f) * vs;
			Vector3F front = base + Vector3F(0.f, 0.f, 1.75f) * vs;
			Vector3F top = base + Vector3F(0.f, 1.75f, 0.f) * vs;
			if((j&1)==0) {
			    right.y = top.y-.1f;
			}
			else {
			    base.x -= .085f * vs;
			}
			
			vrx = 0.725f * (RandomF01() - .5f);
			vry = 1.f  * (RandomF01() + 1.f)  * vy;
			vrz = 0.732f * (RandomF01() - .5f);
			vr = 0.13f * RandomF01();
			
			tetra->addPoint(&base.x);
			hv[0] = vrx + vr;
			hv[1] = vry;
			hv[2] = vrz - vr;
			hv+=3;
			tetra->addPoint(&right.x);
			hv[0] = vrx - vr;
			hv[1] = vry;
			hv[2] = vrz + vr;
			hv+=3;
			tetra->addPoint(&top.x);
			hv[0] = vrx + vr;
			hv[1] = vry;
			hv[2] = vrz + vr;
			hv+=3;
			tetra->addPoint(&front.x);
			hv[0] = vrx - vr;
			hv[1] = vry;
			hv[2] = vrz - vr;
			hv+=3;

			unsigned b = (j * GRDW + i) * 4;
			tetra->addTetrahedron(b, b+1, b+2, b+3);		
		}
		vy = -vy;
	}
	
	tetra->setTotalMass(100.f);
	world->addTetrahedronSystem(tetra);
}

void DynamicWorldInterface::draw(TetrahedronSystem * tetra)
{
    glEnable(GL_DEPTH_TEST);
	glColor3f(0.6f, 0.62f, 0.6f);
    
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	
    glEnableClientState(GL_VERTEX_ARRAY);

	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)tetra->hostX());
	glDrawElements(GL_TRIANGLES, tetra->numTriangleFaceVertices(), GL_UNSIGNED_INT, tetra->hostTriangleIndices());

	glDisableClientState(GL_VERTEX_ARRAY);
	
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	
	glColor3f(0.31f, 0.32f, 0.4f);
	
    glEnableClientState(GL_VERTEX_ARRAY);

	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)tetra->hostX());
	glDrawElements(GL_TRIANGLES, tetra->numTriangleFaceVertices(), GL_UNSIGNED_INT, tetra->hostTriangleIndices());

	glDisableClientState(GL_VERTEX_ARRAY);
}

void DynamicWorldInterface::draw(CudaDynamicWorld * world)
{
    const unsigned nobj = world->numObjects();
    if(nobj<1) return;
    
    unsigned i;
    for(i=0; i< nobj; i++) {
        CudaTetrahedronSystem * tetra = world->tetradedron(i);
        if(tetra) draw(tetra);
    }
}

void DynamicWorldInterface::draw(CudaDynamicWorld * world, GeoDrawer * drawer)
{
    draw(world);
    glDisable(GL_DEPTH_TEST);
	
#if DRAW_BPH_PAIRS
    showOverlappingPairs(world, drawer);
#endif

#if DRAW_BVH_HASH
	showBvhHash(world, drawer);
#endif

#if DRAW_NPH_CONTACT
	showContacts(world, drawer);
#endif
}

void DynamicWorldInterface::drawFaulty(CudaDynamicWorld * world, GeoDrawer * drawer)
{
    glDisable(GL_DEPTH_TEST);
    draw(world);
    showFaultyPair(world, drawer);
}

#if DRAW_BPH_PAIRS
void DynamicWorldInterface::showOverlappingPairs(CudaDynamicWorld * world, GeoDrawer * drawer)
{
    CudaBroadphase * broadphase = world->broadphase();
    // std::cout<<" num overlapping pairs "<<broadphase->numUniquePairs()<<" ";
	
    const unsigned cacheLength = broadphase->pairCacheLength();
	if(cacheLength < 1) return;
	
	Aabb * boxes = (Aabb *)broadphase->hostAabb();
	Aabb abox;
	BoundingBox ab, bb;
	unsigned i;
	drawer->setColor(0.f, 0.1f, 0.3f);
	
	unsigned * pc = (unsigned *)broadphase->hostPairCache();
	
	unsigned objectI;
	for(i=0; i < broadphase->numOverlappingPairs(); i++) {
	    objectI = extractObjectInd(pc[i * 2]);
	    abox = boxes[broadphase->objectStart(objectI) + extractElementInd(pc[i * 2])];
	    
		bb.setMin(abox.low.x, abox.low.y, abox.low.z);
		bb.setMax(abox.high.x, abox.high.y, abox.high.z);
	    
	    objectI = extractObjectInd(pc[i * 2 + 1]);
	    abox = boxes[broadphase->objectStart(objectI) + extractElementInd(pc[i * 2 + 1])];
	    
	    ab.setMin(abox.low.x, abox.low.y, abox.low.z);
		ab.setMax(abox.high.x, abox.high.y, abox.high.z);
		
		drawer->arrow(bb.center(), ab.center());
		
		bb.expandBy(ab);
		
		// drawer->boundingBox(bb);
	}
}
#endif

#if DRAW_BVH_HASH
void DynamicWorldInterface::showBvhHash(CudaDynamicWorld * world, GeoDrawer * drawer)
{
    CudaBroadphase * broadphase = world->broadphase();
    const unsigned n = broadphase->numObjects();
    unsigned i;
    for(i=0; i< n; i++)
        showBvhHash(broadphase->object(i), drawer);
}

void DynamicWorldInterface::showBvhHash(CudaLinearBvh * bvh, GeoDrawer * drawer)
{
    const unsigned n = bvh->numLeafNodes();
	Aabb * boxes = (Aabb *)bvh->hostLeafBox();
	
	KeyValuePair * bvhHash = (KeyValuePair *)bvh->hostLeafHash();
	
	float red;
	Vector3F p, q;
	for(unsigned i=1; i < n; i++) {
		red = (float)i/(float)n;
		
		glColor3f(red, 1.f - red, 0.f);
		Aabb a0 = boxes[bvhHash[i-1].value];
		p.set(a0.low.x * 0.5f + a0.high.x * 0.5f, a0.low.y * 0.5f + a0.high.y * 0.5f + 0.2f, a0.low.z * 0.5f + a0.high.z * 0.5f);
		Aabb a1 = boxes[bvhHash[i].value];
		q.set(a1.low.x * 0.5f + a1.high.x * 0.5f, a1.low.y * 0.5f + a1.high.y * 0.5f + 0.2f, a1.low.z * 0.5f + a1.high.z * 0.5f);
        
		drawer->arrow(p, q);
	}
}
#endif

#if DRAW_NPH_CONTACT
void DynamicWorldInterface::showContacts(CudaDynamicWorld * world, GeoDrawer * drawer)
{
    CudaNarrowphase * narrowphase = world->narrowphase();
    const unsigned n = narrowphase->numContacts();
    // std::cout<<" num contact pairs "<<n<<"\n";
	if(n<1) return;
	
	SimpleContactSolver * solver = world->contactSolver();
	// if(solver->numContacts() != n) return;
	
	storeModels(narrowphase);
	
	Vector3F * tetPnt = (Vector3F *)m_tetPnt->data();
	unsigned * tetInd = (unsigned *)m_tetInd->data();
	unsigned * pntOffset = (unsigned *)m_pointStarts->data();
	unsigned * indOffset = (unsigned *)m_indexStarts->data();
	
    m_contactPairs->create(n * 8);
    narrowphase->getContactPairs(m_contactPairs);
    
    unsigned * c = (unsigned *)m_contactPairs->data();
    unsigned i, j;
    glColor3f(0.4f, 0.9f, 0.6f);
	Vector3F dst, cenA, cenB;
    
	CUDABuffer * bodyPair = solver->contactPairHashBuf();
	m_pairsHash->create(bodyPair->bufferSize());
	bodyPair->deviceToHost(m_pairsHash->data(), m_pairsHash->bufferSize());
	
	m_linearVelocity->create(n * 2 * 12);
	solver->deltaLinearVelocityBuf()->deviceToHost(m_linearVelocity->data(), 
	    m_linearVelocity->bufferSize());
	Vector3F * linVel = (Vector3F *)m_linearVelocity->data();
	
	m_angularVelocity->create(n * 2 * 12);
	solver->deltaAngularVelocityBuf()->deviceToHost(m_angularVelocity->data(), 
	    m_angularVelocity->bufferSize());
	Vector3F * angVel = (Vector3F *)m_angularVelocity->data();
	
	m_contact->create(n * 48);
	narrowphase->contactBuffer()->deviceToHost(m_contact->data(), m_contact->bufferSize());
	ContactData * contact = (ContactData *)m_contact->data();

	Vector3F N;

	bool isA;
	unsigned iPairA, iBody, iPair;
	unsigned * bodyAndPair = (unsigned *)m_pairsHash->data();
	bool converged;
	for(i=0; i < n * 2; i++) {

	    iBody = bodyAndPair[i*2];
	    iPair = bodyAndPair[i*2+1];
	    
	    // std::cout<<"body "<<iBody<<" pair "<<iPair<<"\n";
	    
	    iPairA = iPair * 2;
// left or right
        isA = (iBody == c[iPairA]);

	    cenA = tetrahedronCenter(tetPnt, tetInd, pntOffset, indOffset, iBody);
 
	    //cenB = cenA + angVel[i];
	    
	    //glColor3f(0.1f, 0.7f, 0.3f);
	    //m_drawer->arrow(cenA, cenB);
	    
	    ContactData & cd = contact[iPair];
	    float4 sa = cd.separateAxis;
	    N.set(sa.x, sa.y, sa.z);
	    N.reverse();
	    N.normalize();

	    if(isA) {
// show contact normal for A
		    cenB = cenA + Vector3F(cd.localA.x, cd.localA.y, cd.localA.z);
		    drawer->setColor(0.f, .3f, .9f);
		    drawer->arrow(cenB, cenB + N);
		}

		glColor3f(0.73f, 0.68f, 0.1f);
		drawer->arrow(cenA, cenA + linVel[i]);
		
		glColor3f(0.1f, 0.68f, 0.72f);
		drawer->arrow(cenA, cenA + angVel[i]);
	}
}
#endif

void DynamicWorldInterface::printConstraint(SimpleContactSolver * solver, unsigned n)
{
    unsigned * pairs = (unsigned *)m_pairCache->data();
	
	m_constraint->create(n * 64);
    solver->constraintBuf()->deviceToHost(m_constraint->data(), n * 64);
	ContactConstraint * constraint = (ContactConstraint *)m_constraint->data();
    
	unsigned i;
	BarycentricCoordinate coord;
	float sum;
	for(i=0; i < n; i++) {
	    std::cout<<(boost::str(boost::format("constraint[%1%] (%2%,%3%)\n") % i % pairs[i*2] % pairs[i*2+1]));
        std::cout<<(boost::str(boost::format("n (%1%,%2%,%3%)\n") % constraint[i].normal.x % constraint[i].normal.y % constraint[i].normal.z));
        coord = constraint[i].coordA;
	    std::cout<<(boost::str(boost::format("ca (%1%,%2%,%3%,%4%) ") % coord.x % coord.y % coord.z % coord.w));
	    coord = constraint[i].coordB;
	    std::cout<<(boost::str(boost::format("cb (%1%,%2%,%3%,%4%)\n") % coord.x % coord.y % coord.z % coord.w));
	    std::cout<<(boost::str(boost::format("minv %1%\n") % constraint[i].Minv));
	    std::cout<<(boost::str(boost::format("relvel %1%\n") % constraint[i].relVel));
	}
}

bool DynamicWorldInterface::checkConstraint(SimpleContactSolver * solver, unsigned n)
{
    unsigned * pairs = (unsigned *)m_pairCache->data();
	
	m_constraint->create(n * 64);
    solver->constraintBuf()->deviceToHost(m_constraint->data(), n * 64);
	ContactConstraint * constraint = (ContactConstraint *)m_constraint->data();
    
	unsigned i;
	BarycentricCoordinate coord;
	float sum;
	for(i=0; i < n; i++) {
	    if(IsNan(constraint[i].normal.x) || IsNan(constraint[i].normal.y) || IsNan(constraint[i].normal.z)) {
	        std::cout<<"invalide normal["<<i<<"]\n";
	        m_faultyPair[0] = pairs[i*2];
	        m_faultyPair[1] = pairs[i*2+1];
	        return false;
	    }
	    
	    coord = constraint[i].coordA;
	    sum = coord.x + coord.y + coord.z + coord.w;
	    if(sum > 1.1f || sum < .9f) {
	        std::cout<<"invalid coord A["<<i<<"] "<<coord.x<<","<<coord.y<<","<<coord.z<<","<<coord.z<<"\n";
	        m_faultyPair[0] = pairs[i*2];
	        m_faultyPair[1] = pairs[i*2+1];
	        return false;
	    }
	    coord = constraint[i].coordB;
	    sum = coord.x + coord.y + coord.z + coord.w;
	    if(sum > 1.1f || sum < .9f) {
	        std::cout<<"invalide coord B["<<i<<"] "<<coord.x<<","<<coord.y<<","<<coord.z<<","<<coord.z<<"\n";
	        m_faultyPair[0] = pairs[i*2];
	        m_faultyPair[1] = pairs[i*2+1]; 
	        return false;
	    }
	    
	    if(IsNan(constraint[i].Minv) || IsInf(constraint[i].Minv)) {
	        std::cout<<"pair["<<i<<"] ("<<pairs[i*2]<<","<<pairs[i*2+1]<<")\n";
	        std::cout<<"invalid minv["<<i<<"] "<<constraint[i].Minv<<"\n";
	        m_faultyPair[0] = pairs[i*2];
	        m_faultyPair[1] = pairs[i*2+1];
	        return false;
	    }
	    
	    if(IsNan(constraint[i].relVel) || IsInf(constraint[i].relVel)) {
	        std::cout<<"invalid relvel"<<"["<<i<<"] "<<constraint[i].relVel<<"\n";
	        m_faultyPair[0] = pairs[i*2];
	        m_faultyPair[1] = pairs[i*2+1];
	        return false;
	    }
	}
	
	return true;
}

bool DynamicWorldInterface::checkContact(unsigned n)
{
    ContactData * contact = (ContactData *)m_contact->data();
    unsigned * pairs = (unsigned *)m_pairCache->data();
    Vector3F sa;
	unsigned i;
	for(i=0; i<n; i++) {
	    sa.set(contact[i].separateAxis.x, contact[i].separateAxis.y, contact[i].separateAxis.z);
	    if(IsNan(sa.x) || IsNan(sa.y) || IsNan(sa.z)) {
	        std::cout<<"("<<pairs[i*2]<<","<<pairs[i*2+1]<<")separate axis is Nan\n";
	        m_faultyPair[0] = pairs[i*2];
	        m_faultyPair[1] = pairs[i*2+1];
	        return false;
	    }
	    if(sa.length() < 1e-9) {
	        std::cout<<"("<<pairs[i*2]<<","<<pairs[i*2+1]<<")separate axis is close to zero\n";
	        m_faultyPair[0] = pairs[i*2];
	        m_faultyPair[1] = pairs[i*2+1];
	        return false;
	    }
	}
	return true;
}

bool DynamicWorldInterface::checkDegenerated(unsigned n)
{
    Vector3F * tetPnt = (Vector3F *)m_tetPnt->data();
	unsigned * tetInd = (unsigned *)m_tetInd->data();
	unsigned * pntOffset = (unsigned *)m_pointStarts->data();
	unsigned * indOffset = (unsigned *)m_indexStarts->data();
	unsigned * pairs = (unsigned *)m_pairCache->data();
	Vector3F p[4];
	Matrix44F mat;
	unsigned i;
	for(i=0; i < n; i++) {
	    tetrahedronPoints(p, tetPnt, tetInd, pntOffset, indOffset, pairs[i*2]);
	    if(isTetrahedronDegenerated(p)) {
	        std::cout<<"degenerated tetrahedron ("<<pairs[i*2]<<"),"<<pairs[i*2+1]<<" ";
	        std::cout<<"det "<<determinantTetrahedron(mat, p[0], p[1], p[2], p[3])<<"\n";
	        m_faultyPair[0] = pairs[i*2];
	        m_faultyPair[1] = pairs[i*2+1];
	        return false;
	    }
	    tetrahedronPoints(p, tetPnt, tetInd, pntOffset, indOffset, pairs[i*2+1]);
	    if(isTetrahedronDegenerated(p)) {
	        std::cout<<"degenerated tetrahedron "<<pairs[i*2]<<",("<<pairs[i*2+1]<<") ";
	        std::cout<<"det "<<determinantTetrahedron(mat, p[0], p[1], p[2], p[3])<<"\n";
	        m_faultyPair[0] = pairs[i*2];
	        m_faultyPair[1] = pairs[i*2+1];
	        return false;
	    }
	}
    return true;
}

void DynamicWorldInterface::printContact(unsigned n)
{
    ContactData * contact = (ContactData *)m_contact->data();
    unsigned * pairs = (unsigned *)m_pairCache->data();
    Vector3F sa;
	unsigned i;
	for(i=0; i<n; i++) {
	    sa.set(contact[i].separateAxis.x, contact[i].separateAxis.y, contact[i].separateAxis.z);
	    std::cout<<"contact["<<i<<"] ("<<pairs[i*2]<<","<<pairs[i*2+1]<<")\n sa"<<sa<<"\n";
	    sa.set(contact[i].localA.x, contact[i].localA.y, contact[i].localA.z);
	    std::cout<<"la "<<sa<<" ";
	    sa.set(contact[i].localB.x, contact[i].localB.y, contact[i].localB.z);
	    std::cout<<"lb "<<sa<<"\n";
	    std::cout<<"toi "<<contact[i].timeOfImpact<<"\n";
	}
}

bool DynamicWorldInterface::checkConvergent(SimpleContactSolver * solver, unsigned n)
{
    const unsigned njacobi = solver->numIterations();
	m_deltaJ->create(n * njacobi * 4);
	solver->deltaJBuf()->deviceToHost(m_deltaJ->data(), m_deltaJ->bufferSize());
	float * dJ = (float *)m_deltaJ->data();
	
	unsigned * pairs = (unsigned *)m_pairCache->data();
    
	unsigned i, j;
	int converged;
	float lastJ, curJ;
	for(i=0; i<n; i++) {
	    converged = 1;
	    lastJ = dJ[i * njacobi];
	    if(lastJ < 1e-3) converged = 0;
	    for(j=1; j< njacobi; j++) {
	        curJ = dJ[i * njacobi + j];
	        if(curJ < 0.f) curJ = -curJ;
	        if(curJ >= lastJ && curJ > 1e-3) {
                converged = 0;
                break;
            }
            lastJ = curJ;
        }
        if(!converged) {
            std::cout<<(boost::str(boost::format("contact(%1%,%2%) not convergent\n") % pairs[i*2] % pairs[i*2+1]));
            m_faultyPair[0] = pairs[i*2];
	        m_faultyPair[1] = pairs[i*2+1];
	        
	        for(j=0; j< njacobi; j++)
	            std::cout<<(boost::str(boost::format("dJ[%1%] %2%\n") % j % dJ[i * njacobi + j]));
	        return false;
        }
	}
   
    return true;
}

bool DynamicWorldInterface::verifyData(CudaDynamicWorld * world)
{
    CudaNarrowphase * narrowphase = world->narrowphase();
    const unsigned n = narrowphase->numContacts();
    if(n<1) return true;
    
    SimpleContactSolver * solver = world->contactSolver();
	
	storeModels(narrowphase);
	
	m_pairCache->create(n * 8);
	CUDABuffer * pairbuf = narrowphase->contactPairsBuffer();
	pairbuf->deviceToHost(m_pairCache->data(), m_pairCache->bufferSize());
	
	if(!checkDegenerated(n)) {
	    std::cout<<"degenerated tetrahedron\n";
	    printFaultPair(world);
	    return false;
	}
	
	m_contact->create(n * 48);
	narrowphase->contactBuffer()->deviceToHost(m_contact->data(), m_contact->bufferSize());
	
	if(!checkContact(n)) {
	    std::cout<<"invalid contact\n";
	    printContact(n);
	    return false;
	}
	
	if(!checkConstraint(solver, n)) {
	    std::cout<<"invalid constraint\n";
	    printContactPairHash(solver, n);
	    return false;
	}
	
    if(DynGlobal::CheckConvergence) {
        if(!checkConvergent(solver, n)) {
            std::cout<<"not convergent\n";
            printContact(n);
            printConstraint(solver, n);
            printContactPairHash(solver, n);
            return false;
        }
    }
    
	CUDABuffer * bodyPair = solver->contactPairHashBuf();
	m_pairsHash->create(bodyPair->bufferSize());
	bodyPair->deviceToHost(m_pairsHash->data(), m_pairsHash->bufferSize());
	unsigned * bodyAndPair = (unsigned *)m_pairsHash->data();
    
    m_linearVelocity->create(n * 2 * 12);
	solver->deltaLinearVelocityBuf()->deviceToHost(m_linearVelocity->data(), 
	    m_linearVelocity->bufferSize());
	Vector3F * linVel = (Vector3F *)m_linearVelocity->data();
	
	m_angularVelocity->create(n * 2 * 12);
	solver->deltaAngularVelocityBuf()->deviceToHost(m_angularVelocity->data(), 
	    m_angularVelocity->bufferSize());
	Vector3F * angVel = (Vector3F *)m_angularVelocity->data();
	
	Vector3F N;
	unsigned iPairA, iBody, iPair;
	unsigned i;
	for(i=0; i < n * 2; i++) {
	    iBody = bodyAndPair[i*2];
	    iPair = bodyAndPair[i*2+1];

	    if(IsNan(linVel[i].x) || IsNan(linVel[i].y) || IsNan(linVel[i].z)) {
	        std::cout<<"delta linear velocity is Nan\n";
	        return false;
	    }
	    
	    if(IsNan(angVel[i].x) || IsNan(angVel[i].y) || IsNan(angVel[i].z)) {
	        std::cout<<"delta angular velocity is Nan\n";
	        return false;
	    }
	}
    
    return true;
}

void DynamicWorldInterface::printContactPairHash(SimpleContactSolver * solver, unsigned numContacts)
{
    unsigned i;
	
    m_mass->create(numContacts * 2 * 4);
    CUDABuffer * splitMassBuf = solver->splitInverseMassBuf();
    splitMassBuf->deviceToHost(m_mass->data(), m_mass->bufferSize());
	float * mass = (float *)m_mass->data();
	
	std::cout<<" mass:\n";
	for(i=0; i < numContacts * 2; i++)
		std::cout<<" "<<i<<" ("<<mass[i]<<")\n";
	
	CUDABuffer * splitbuf = solver->bodySplitLocBuf();
	m_split->create(splitbuf->bufferSize());
	splitbuf->deviceToHost(m_split->data(), m_split->bufferSize());
	
	unsigned * split = (unsigned *)m_split->data();
	std::cout<<" split pairs:\n";
	for(i=0; i < numContacts; i++) {
		std::cout<<" "<<i<<" ("<<split[i*2]<<","<<split[i*2+1]<<")\n";
	}
    
    unsigned * pairs = (unsigned *)m_pairCache->data();
	std::cout<<" body(loc)(mass)-body(loc)(mass) pair:\n";
	for(i=0; i < numContacts; i++) {
		std::cout<<" "<<i<<" ("<<pairs[i*2]<<"("<<split[i*2]<<")("<<mass[split[i*2]]<<"),"<<pairs[i*2+1]<<"("<<split[i*2+1]<<")("<<mass[split[i*2+1]]<<"))\n";
	}
	
	CUDABuffer * pairbuf = solver->contactPairHashBuf();
	m_pairsHash->create(pairbuf->bufferSize());
	pairbuf->deviceToHost(m_pairsHash->data(), m_pairsHash->bufferSize());
	
	unsigned * bodyPair = (unsigned *)m_pairsHash->data();
	
	std::cout<<" body-contact hash:\n";
	for(i=0; i < numContacts * 2; i++) {
		std::cout<<" "<<i<<" ("<<bodyPair[i*2]<<","<<bodyPair[i*2+1]<<")\n";
	}
}

void DynamicWorldInterface::storeModels(CudaNarrowphase * narrowphase)
{
    CUDABuffer * pnts = narrowphase->objectBuffer()->m_pos;
	CUDABuffer * inds = narrowphase->objectBuffer()->m_ind;
	CUDABuffer * pointStarts = narrowphase->objectBuffer()->m_pointCacheLoc;
	CUDABuffer * indexStarts = narrowphase->objectBuffer()->m_indexCacheLoc;
	
	m_tetPnt->create(pnts->bufferSize());
	m_tetInd->create(inds->bufferSize());
	m_pointStarts->create(pointStarts->bufferSize());
	m_indexStarts->create(indexStarts->bufferSize());
	
	pnts->deviceToHost(m_tetPnt->data(), m_tetPnt->bufferSize());
	inds->deviceToHost(m_tetInd->data(), m_tetInd->bufferSize());
	pointStarts->deviceToHost(m_pointStarts->data(), m_pointStarts->bufferSize());
	indexStarts->deviceToHost(m_indexStarts->data(), m_indexStarts->bufferSize());
}

void DynamicWorldInterface::printFaultPair(CudaDynamicWorld * world)
{
    CudaNarrowphase * narrowphase = world->narrowphase();
    const unsigned n = narrowphase->numContacts();
    if(n<1) return; 
    
    storeModels(narrowphase);
    
    Vector3F * tetPnt = (Vector3F *)m_tetPnt->data();
	unsigned * tetInd = (unsigned *)m_tetInd->data();
	unsigned * pntOffset = (unsigned *)m_pointStarts->data();
	unsigned * indOffset = (unsigned *)m_indexStarts->data();
	
	Vector3F p[4];
	tetrahedronPoints(p, tetPnt, tetInd, pntOffset, indOffset, m_faultyPair[0]);

	int i;
	for(i=0; i<4; i++) std::cout<<"pa"<<p[i]<<" ";
	
	tetrahedronPoints(p, tetPnt, tetInd, pntOffset, indOffset, m_faultyPair[1]);
	
	for(i=0; i<4; i++) std::cout<<"pb"<<p[i]<<" ";
}

void DynamicWorldInterface::showFaultyPair(CudaDynamicWorld * world, GeoDrawer * drawer)
{
    CudaNarrowphase * narrowphase = world->narrowphase();
    const unsigned n = narrowphase->numContacts();
    if(n<1) return; 
    
    storeModels(narrowphase);
    
    Vector3F * tetPnt = (Vector3F *)m_tetPnt->data();
	unsigned * tetInd = (unsigned *)m_tetInd->data();
	unsigned * pntOffset = (unsigned *)m_pointStarts->data();
	unsigned * indOffset = (unsigned *)m_indexStarts->data();
	
	Vector3F p[4];
	tetrahedronPoints(p, tetPnt, tetInd, pntOffset, indOffset, m_faultyPair[0]);
	glColor3f(1.f, 0.f, 0.f);
	drawer->arrow(p[0], p[1]);
	drawer->arrow(p[0], p[2]);
	drawer->arrow(p[0], p[3]);
	drawer->arrow(p[1], p[2]);
	drawer->arrow(p[2], p[3]);
	drawer->arrow(p[3], p[1]);
	tetrahedronPoints(p, tetPnt, tetInd, pntOffset, indOffset, m_faultyPair[1]);
	drawer->arrow(p[0], p[1]);
	drawer->arrow(p[0], p[2]);
	drawer->arrow(p[0], p[3]);
	drawer->arrow(p[1], p[2]);
	drawer->arrow(p[2], p[3]);
	drawer->arrow(p[3], p[1]);
	drawer->arrow(tetrahedronCenter(tetPnt, tetInd, pntOffset, indOffset, m_faultyPair[0]),
	             tetrahedronCenter(tetPnt, tetInd, pntOffset, indOffset, m_faultyPair[1]));
}

#if DRAW_BVH_HIERARCHY
void DynamicWorldInterface::showBvhHierarchy(CudaLinearBvh * bvh)
{
	bvh->getRootNodeIndex(&m_hostRootNodeInd);
	
	const unsigned numInternal = bvh->numInternalNodes();
	
	Aabb * internalBoxes = (Aabb *)bvh->hostInternalAabb();
	
	KeyValuePair * leafHash = (KeyValuePair *)bvh->hostLeafHash();
	
	m_displayLeafAabbs->create(bvh->numLeafNodes() * sizeof(Aabb));
	bvh->getLeafAabbs(m_displayLeafAabbs);
	Aabb * leafBoxes = (Aabb *)m_displayLeafAabbs->data();
	
	m_internalChildIndices->create(bvh->numInternalNodes() * sizeof(int2));
	bvh->getInternalChildIndex(m_internalChildIndices);
	int2 * internalNodeChildIndices = (int2 *)m_internalChildIndices->data();
	
	m_displayInternalDistance->create(bvh->numInternalNodes() * sizeof(int));
	bvh->getInternalDistances(m_displayInternalDistance);
	int * levels = (int *)m_displayInternalDistance->data();
	
	BoundingBox bb;
	
	int stack[128];
	stack[0] = m_hostRootNodeInd;
	int stackSize = 1;
	int maxStack = 1;
	int touchedLeaf = 0;
	int touchedInternal = 0;
	while(stackSize > 0) {
		int internalOrLeafNodeIndex = stack[ stackSize - 1 ];
		stackSize--;
		
		int isLeaf = isLeafNode(internalOrLeafNodeIndex);	//Internal node if false
		uint bvhNodeIndex = getIndexWithInternalNodeMarkerRemoved(internalOrLeafNodeIndex);
		
		int bvhRigidIndex = (isLeaf) ? leafHash[bvhNodeIndex].value : -1;
		
		Aabb bvhNodeAabb = (isLeaf) ? leafBoxes[bvhRigidIndex] : internalBoxes[bvhNodeIndex];

		{
			if(isLeaf) {
				glColor3f(.5, 0., 0.);
				bb.setMin(bvhNodeAabb.low.x, bvhNodeAabb.low.y, bvhNodeAabb.low.z);
				bb.setMax(bvhNodeAabb.high.x, bvhNodeAabb.high.y, bvhNodeAabb.high.z);
				m_drawer->boundingBox(bb);

				touchedLeaf++;
			}
			else {
				glColor3f(.5, .65, 0.);
				
				if(levels[bvhNodeIndex] > m_displayLevel) continue;
				bb.setMin(bvhNodeAabb.low.x, bvhNodeAabb.low.y, bvhNodeAabb.low.z);
				bb.setMax(bvhNodeAabb.high.x, bvhNodeAabb.high.y, bvhNodeAabb.high.z);
				m_drawer->boundingBox(bb);

				touchedInternal++;
				if(stackSize + 2 > 128)
				{
					//Error
				}
				else
				{
				    stack[ stackSize ] = internalNodeChildIndices[bvhNodeIndex].x;
					stackSize++;
					stack[ stackSize ] = internalNodeChildIndices[bvhNodeIndex].y;
					stackSize++;
					
					if(stackSize > maxStack) maxStack = stackSize;
				}
			}
		}
	}	
}
#endif
