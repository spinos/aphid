#include "SkeletonSystem.h"
#include <SkeletonJoint.h>
SkeletonSystem::SkeletonSystem() {}
SkeletonSystem::~SkeletonSystem() 
{
    clear();
}

void SkeletonSystem::clear()
{
    std::vector<SkeletonJoint *>::iterator it = m_joints.begin();
    for(; it != m_joints.end(); ++it) delete (*it); 
    m_joints.clear();
}

void SkeletonSystem::addJoint(SkeletonJoint * j)
{
    m_joints.push_back(j);
    j->setIndex(numJoints() - 1);
}

unsigned SkeletonSystem::numJoints() const
{
    return m_joints.size();
}
   
SkeletonJoint * SkeletonSystem::joint(unsigned idx) const
{
    return m_joints[idx];
}

SkeletonJoint * SkeletonSystem::selectJoint(const Ray & ray) const
{
    std::vector<SkeletonJoint *>::const_iterator it = m_joints.begin();
	for(; it != m_joints.end(); ++it) {
		if((*it)->intersect(ray)) return *it;
	}
	return m_joints[0];
}


