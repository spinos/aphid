/*
 *  ElasticRodAttachmentConstraint.h
 *  
 *
 *  Created by jian zhang on 1/7/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_PBD_ELASTIC_ROD_ATTACHMENT_CONSTRAINT_H
#define APH_PBD_ELASTIC_ROD_ATTACHMENT_CONSTRAINT_H

#include "ElasticRodBendAndTwistConstraint.h"

namespace aphid {

namespace pbd {

class ParticleData;

class ElasticRodAttachmentConstraint : public ElasticRodBendAndTwistConstraint {
	
	Vector3F m_p0, m_p1, m_g0;
	
public:
	ElasticRodAttachmentConstraint();
	
	virtual ConstraintType getConstraintType() const;
    
	bool initConstraint(SimulationContext * model, const int pA, const int pB, const int pC,
                                    const int pD, const int pE);
									
	bool solvePositionConstraint(ParticleData* part, ParticleData* ghost);
	
	void updateConstraint(ParticleData* part, ParticleData* ghost);
	
protected:

};

}
}

#endif

