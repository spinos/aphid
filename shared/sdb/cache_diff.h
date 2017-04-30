/*
 *  cache_diff.h
 *  
 *  cache_diff(c, q)
 *  record element in q not in c, store result in c
 *  clear q
 *	return size of c
 *  Created by jian zhang on 1/15/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_SDB_CACHE_DIFF_H
#define APH_SDB_CACHE_DIFF_H

namespace aphid {

namespace sdb {

template<typename T>
inline int cache_diff(T & c, T & q)
{
	T d;
	
	q.begin();
	while(!q.end() ) {
		if(!c.findKey(q.key() ) ) {
			d.insert(q.key() );
		}
		q.next();
	}
	
	c.clear();
	q.clear();
	
	if(d.size() < 1) {
		return 0;
	}
	
	int n = 0;
	d.begin();
	while(!d.end() ) {
		c.insert(d.key() );
		n++;
		
		d.next();
	}
	
	return n;
}

}

}
#endif