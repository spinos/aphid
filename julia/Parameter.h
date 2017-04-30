#ifndef JUL_PARAMETER_H
#define JUL_PARAMETER_H

#include <string>
#include <vector>

namespace jul {

class Parameter {

	std::string m_inFileName;
	std::string m_outFileName;
	int m_cellSize;
	
public:
	enum OperationFlag {
		kUnknown = 0,
		kHelp = 1,
		kGenerate = 2,
		kBuildTree = 3,
		kInitialize = 4,
		kInsert = 5,
		kRemove = 6
	};
	
	Parameter(int argc, char *argv[]);
	virtual ~Parameter();
	
	bool isValid() const;
	static void PrintHelp();
	
	OperationFlag operation() const;
	
	const std::string & inFileName() const;
	const std::string & outFileName() const;
	const int & cellSize() const;
	
protected:

private:
	OperationFlag m_opt;
};

}
#endif        //  #ifndef LFPARAMETER_H

