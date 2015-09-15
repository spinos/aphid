#ifndef _S_HELPER_H
#define _S_HELPER_H
#include <fstream>
#include <string>
#include <vector>


class SHelper
{
public:
	SHelper() {}
	~SHelper() {}
	
	static void divideByFirstSpace(std::string& ab2a, std::string& b);
	static void trimByFirstSpace(std::string& res);
	static void getTime(std::string& log);
	static void cutByFirstDot(std::string& res);
	static void cutByLastDot(std::string& res);
	static void cutByLastSlash(std::string& res);
	static void changeFrameNumber(std::string& res, int frame);
	static void changeFrameNumber(std::string& res, int frame, int padding);
	static int safeConvertToInt(const double a);
	static int getFrameNumber(std::string& name);
	static int compareFilenameExtension(std::string& name, const char* other);
	static void setFrameNumberAndExtension(std::string& name, int frame, const char* extension);
	static int getFrameNumber(const std::string& name);
	static void removeFilenameExtension(std::string& name);
	static void changeFilenameExtension(std::string& name, const char* ext);
	static void validateFilePath(std::string& name);
	static void replacefilename(std::string& res, std::string& toreplace);
	static void findLastAndReplace(std::string& res, const char *tofind, const char *toreplace);
	static void cutfilepath(std::string& res);
	static void changeFrameNumberFistDot4Digit(std::string& res, int frame);
	static char isInArrayDividedBySpace(const char* handle, const char* array);
	static void filenameWithoutPath(std::string& res);
	static void protectComma(std::string& res);
	static void ribthree(std::string& res);
	static int findPartBeforeChar(std::string& full, std::string& frag, int start, char sep);
	static void protectCommaFree(std::string& res);
	static void endNoReturn(std::string& res);
	static std::string getParentName(const std::string& name);
	static std::string getLastName(const std::string& name);
	static std::string getHighestParentName(std::string& name);
	static void getHierarchy(const char *name, std::vector<std::string> &res);
	static void listAllNames(std::string& name, std::vector<std::string>& all);
	static char hasParent(std::string& name);
	static void listParentNames(const std::string& name, std::vector<std::string>& parents);
	static void pathToFilename(std::string& name);
	static void noColon(std::string& name);
	static void pathDosToUnix(std::string& name);
	static void pathUnixToDos(std::string& name);
	static void removePrefix(std::string& name);
	static void removeNodeName(std::string& name);
	static void behead(std::string& name, const std::string& head);
	static void stripAll(std::string& name);
	static std::string afterLastUnderscore(const std::string &res);
	static char fuzzyMatch(std::string &one,std::string &another);
	static char removeAnyNamespace(std::string &name, const char * separator = "|");
	static char replaceAnyNamespace(std::string &name,std::string &_namespace);
	static char fuzzyMatchNamespace(std::string &one,std::string &another);
	static std::string removeNamespace(const std::string &in);
	static int countColons(const std::string &in);
	static void validateUnixPath(std::string& name);
	static char isMatched(const std::string &one, const std::string &another);
	static bool IsPullPath(const std::string & name);
    static std::string ParentPath(const std::string & name, const std::string & separator);
	static std::string Combine(const std::vector<std::string > & names);
	static void Split(const std::string & combined, std::vector<std::string > & names);
	static bool Find(const std::string & name, const std::string & part, bool caseSensitive = true);
    static bool DeduceAttribName(const std::string & hesPath, std::string & attribName);
    static bool GetNamespace(std::string & res);
    static bool GetFirstNamespace(std::string & res);
    static std::string GetParentName(const std::string& name, const std::string & term);
    static std::string GetFollowupName(const std::string& name, const std::string & term);
};
#endif
//:~
