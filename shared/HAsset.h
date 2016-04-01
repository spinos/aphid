/*
 *  HAsset.h
 *  julia
 *
 *  Created by jian zhang on 4/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include <HBase.h>
#include <Boundary.h>

namespace aphid {

template<typename T>
class HAsset : public HBase, public Boundary {

	int m_active;
	
public:
	HAsset(const std::string & name);
	virtual ~HAsset();
	
	virtual char save();
	virtual char load();
	
	void setActive(const int & x);
	const int & isActive() const;
	
protected:

private:
};

template<typename T>
HAsset<T>::HAsset(const std::string & name) :
HBase(name),
m_active(1)
{}

template<typename T>
HAsset<T>::~HAsset()
{}

template<typename T>
void HAsset<T>::setActive(const int & x)
{ m_active = x; }

template<typename T>
const int & HAsset<T>::isActive() const
{ return m_active; }

template<typename T>
char HAsset<T>::save()
{
	if(!hasNamedAttr(".bbx") )
	    addFloatAttr(".bbx", 6);
	writeFloatAttr(".bbx", (float *)&getBBox() );
	
	if(!hasNamedAttr(".elemtyp") )
	    addIntAttr(".elemtyp", 1);
	
	int et = T::ShapeTypeId;
	writeIntAttr(".elemtyp", (int *)&et );
		
	if(!hasNamedAttr(".act") )
		addIntAttr(".act", 1);
		
	writeIntAttr(".act", (int *)&m_active );
	
	return 1;
}

template<typename T>
char HAsset<T>::load()
{
	BoundingBox b;
	readFloatAttr(".bbx", (float *)&b );
	setBBox(b);
	
	readFloatAttr(".act", (float *)&m_active );
	
	return 1;
}

}