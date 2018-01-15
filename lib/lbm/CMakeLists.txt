file (GLOB Lbm_SRC
    "../../shared/lbm/*.h"
    "../../shared/lbm/*.cpp"
)

ADD_LIBRARY (aphidLbm ${LIB_TYP} 
${Lbm_SRC})

SET (LIBRARY_OUTPUT_PATH ../)
