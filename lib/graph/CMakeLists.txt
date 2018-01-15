file (GLOB graph_SRC
	"../../shared/graph/*.h"
    "../../shared/graph/*.cpp"
)

ADD_LIBRARY (aphidGraph ${LIB_TYP}
${graph_SRC})

SET (LIBRARY_OUTPUT_PATH ../)
