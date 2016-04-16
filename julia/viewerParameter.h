#ifndef JUL_VIEWER_PARAM_H
#define JUL_VIEWER_PARAM_H

#include <string>
#include <vector>

namespace jul {

class ViewerParam {

	std::string m_inFileName;
	
public:
	enum OperationFlag {
		kUnknown = 0,
		kHelp = 1,
		kTestVoxel = 2,
		kTestAsset = 3
	};
	
	ViewerParam(int argc, char *argv[]);
	virtual ~ViewerParam();
	
	bool isValid() const;
	static void PrintHelp();
	
	OperationFlag operation() const;
	
	const std::string & inFileName() const;
	std::string operationTitle() const;
	
protected:

private:
	OperationFlag m_opt;
};

}
#endif        //  #ifndef LFViewerParam_H

