/*
 *  selectionUtil.h
 *  opium
 *
 *  Created by jian zhang on 6/3/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <AHelper.h>
class MString;
class MSelectionList;
#include <string>
#include <vector>
#include <map>

namespace aphid {
    
class SelectionHelper : public AHelper {
public:
	SelectionHelper();
	void setBehead(const MString &name);
	void excludeHeads(MDagPathArray &arr);
	
	void getEveryObj(MObjectArray &arr, MObject &root = MObject::kNullObj);
	void getEverything(MDagPathArray &arr, MObject &root = MObject::kNullObj);
	char getAll(MDagPathArray &arr);
	char getUpper(MDagPathArray &arr);
	char getDown(MDagPathArray &arr);
	void getDepnode(MObjectArray &arr);
	char getSingleMesh(MDagPath &mesh, MObject &component);
	char meshFaceById(MObject & mesh, const int count, const int *idx);
	char isEmptySelection(MSelectionList & sels);
	void reportNumActive(MDagPathArray &arr);
    char getSelected(MDagPathArray &active_list);
	std::vector<std::string> _behead_list;
	std::map<std::string,std::string> _nameMap;
	std::map<std::string,std::string> _ignoreMap;
	
	static MObject GetTypedNode(const MSelectionList & sels,
								const MString & typName,
							 MFn::Type fltTyp = MFn::kInvalid);
	
private:
    void getParents(MDagPath & root, MDagPathArray &active_list);
    void getChildren(MDagPath & root, MDagPathArray &active_list);
};

}