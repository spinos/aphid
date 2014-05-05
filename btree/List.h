/*
 *  List.h
 *  btree
 *
 *  Created by jian zhang on 5/5/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include <Entity.h>
#include <vector>
namespace sdb {

template<typename T>
class List : public Entity {
public:
	List(Entity * parent = NULL) : Entity(parent) {}
	
	virtual ~List() {}
	
	int size() const { return m_v.size(); }
	
	void insert(const T & x) {
		//std::cout<<"insert "<<x;
		m_v.push_back(x);
	}
	
	void remove(const T & x) {}
	
	void getValues(std::vector<T> & dst) const {
		typename std::vector<T>::const_iterator it;
		it = m_v.begin();
		for(; it != m_v.end(); ++it) dst.push_back(*it); 
	}
private:
	
private:
	std::vector<T> m_v;
};
} //end namespace sdb