file (GLOB ttg_SRC
    "../../shared/ttg/*.h"
    "../../shared/ttg/*.cpp"
)

ADD_LIBRARY (aphidTtg ${LIB_TYP}
${ttg_SRC})

SET (LIBRARY_OUTPUT_PATH ../)
