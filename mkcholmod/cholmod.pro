TEMPLATE = lib
TARGET = cholmod
CONFIG += staticlib thread release
CONFIG -= qt
CHOLMOD_CHECK = D:/usr/CHOLMOD/Check
CHOLMOD_CHOLESKY = D:/usr/CHOLMOD/Cholesky
CHOLMOD_CORE = D:/usr/CHOLMOD/Core
CHOLMOD_MATRIXOPS = D:/usr/CHOLMOD/MatrixOps
CHOLMOD_PARTITION = D:/usr/CHOLMOD/Partition
CHOLMOD_MODIFY = D:/usr/CHOLMOD/Modify
CHOLMOD_SUPERNODAL = D:/usr/CHOLMOD/Supernodal
INCLUDEPATH += D:/usr/CHOLMOD/Include D:/usr/SuiteSparse_config \
                D:/usr/COLAMD/Include \
                D:/usr/CCOLAMD/Include \
                D:/usr/AMD/Include \
                D:/usr/CAMD/Include

SOURCES       = $$CHOLMOD_CORE/cholmod_aat.c \
                $$CHOLMOD_CORE/cholmod_add.c \
                $$CHOLMOD_CORE/cholmod_band.c \
                $$CHOLMOD_CORE/cholmod_change_factor.c \
                $$CHOLMOD_CORE/cholmod_common.c \
                $$CHOLMOD_CORE/cholmod_complex.c \
                $$CHOLMOD_CORE/cholmod_copy.c \
                $$CHOLMOD_CORE/cholmod_dense.c \
                $$CHOLMOD_CORE/cholmod_error.c \
                $$CHOLMOD_CORE/cholmod_factor.c \
                $$CHOLMOD_CORE/cholmod_memory.c \
                $$CHOLMOD_CORE/cholmod_sparse.c \
                $$CHOLMOD_CORE/cholmod_transpose.c \
                $$CHOLMOD_CORE/cholmod_triplet.c \
                $$CHOLMOD_CHECK/cholmod_check.c \
                $$CHOLMOD_CHECK/cholmod_read.c \
                $$CHOLMOD_CHECK/cholmod_write.c \
                $$CHOLMOD_CHOLESKY/cholmod_amd.c \ 
                $$CHOLMOD_CHOLESKY/cholmod_analyze.c \ 
                $$CHOLMOD_CHOLESKY/cholmod_colamd.c \ 
                $$CHOLMOD_CHOLESKY/cholmod_etree.c \ 
                $$CHOLMOD_CHOLESKY/cholmod_factorize.c \
                $$CHOLMOD_CHOLESKY/cholmod_postorder.c \
                $$CHOLMOD_CHOLESKY/cholmod_rcond.c \
                $$CHOLMOD_CHOLESKY/cholmod_resymbol.c \
                $$CHOLMOD_CHOLESKY/cholmod_rowcolcounts.c \
                $$CHOLMOD_CHOLESKY/cholmod_rowfac.c \
                $$CHOLMOD_CHOLESKY/cholmod_solve.c \
                $$CHOLMOD_CHOLESKY/cholmod_spsolve.c \
                $$CHOLMOD_MATRIXOPS/cholmod_drop.c \
                $$CHOLMOD_MATRIXOPS/cholmod_horzcat.c \
                $$CHOLMOD_MATRIXOPS/cholmod_norm.c \
                $$CHOLMOD_MATRIXOPS/cholmod_scale.c \
                $$CHOLMOD_MATRIXOPS/cholmod_sdmult.c \
                $$CHOLMOD_MATRIXOPS/cholmod_ssmult.c \
                $$CHOLMOD_MATRIXOPS/cholmod_submatrix.c \
                $$CHOLMOD_MATRIXOPS/cholmod_vertcat.c \
                $$CHOLMOD_MATRIXOPS/cholmod_symmetry.c \
                $$CHOLMOD_PARTITION/cholmod_ccolamd.c \
                $$CHOLMOD_PARTITION/cholmod_csymamd.c \
                $$CHOLMOD_PARTITION/cholmod_metis.c \
                $$CHOLMOD_PARTITION/cholmod_nesdis.c \
                $$CHOLMOD_PARTITION/cholmod_camd.c \
                $$CHOLMOD_MODIFY/cholmod_rowadd.c \
                $$CHOLMOD_MODIFY/cholmod_rowdel.c \
                $$CHOLMOD_MODIFY/cholmod_updown.c \
                $$CHOLMOD_SUPERNODAL/cholmod_super_numeric.c \ 
                $$CHOLMOD_SUPERNODAL/cholmod_super_solve.c \
                $$CHOLMOD_SUPERNODAL/cholmod_super_symbolic.c

DEFINES += NDEBUG DLONG
                



