/*
 *  H5IO.h
 *  opium
 *
 *  Created by jian zhang on 10/20/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <HBase.h>
namespace aphid {

class H5IO {
public:
	static void CreateGroup(const std::string & name);
	template<typename T1, typename T2>
	static void SaveData(const std::string & name, T2 * data)
	{
		T1 grp(name);
		grp.save(data);
		grp.close();
	}
	
	template<typename T1, typename T2>
	static void LoadData(const std::string & name, T2 * data)
	{
		T1 grp(name);
		grp.load(data);
		grp.close();
	}
	
	static std::string BeheadName;
	
protected:
	static void H5PathName(std::string & dst);
	
private:
};

}