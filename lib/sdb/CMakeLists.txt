file (GLOB Sdb_SRC
    "../../shared/sdb/*.h"
    "../../shared/sdb/*.cpp"
)

ADD_LIBRARY (aphidSdb ${LIB_TYP}
${Sdb_SRC})

SET (LIBRARY_OUTPUT_PATH ../)
