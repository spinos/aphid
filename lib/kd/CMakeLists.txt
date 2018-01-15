file (GLOB APH_Kd_SRC
	"../../shared/kd/*.h"
    "../../shared/kd/*.cpp"
)

ADD_LIBRARY (aphidKd ${LIB_TYP}
${APH_Kd_SRC})

SET (LIBRARY_OUTPUT_PATH ../)
