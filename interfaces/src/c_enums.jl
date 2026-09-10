# c_enums.jl — Julia mirror of the C types declared in include/smc.h.
#
# The enumerator values (SmcDataType, SmcStructure, SmcPartition,
# SmcDecompression, SmcOrder) live in scripts/coloring_table.jl, the single
# source of truth shared with the header generator, and are already in scope.

# Mirror of the SmcColoringOptions C struct, in the field order of smc.h
# section 2 and of include/smc.h.  All fields are `Cint`, so the C and Julia
# layouts agree with no padding and the struct is `isbits`: it can be
# `unsafe_load`ed from the caller's pointer and returned by value.
# Changing the field order is an ABI break.
struct SmcColoringOptionsC
    structure::Cint   # SmcStructure
    partition::Cint   # SmcPartition
    decompression::Cint   # SmcDecompression
    order::Cint   # SmcOrder
    postprocessing::Cint   # 0/1 — neutral color 0 where possible
    symmetric_pattern::Cint   # 0/1 — assert the sparsity pattern is symmetric
    index_base::Cint   # 0 or 1 — index base of colptr / rowval / groups
    dtype::Cint   # SmcDataType — element type of compress/decompress
end

# Also returned whenever a NULL options pointer is passed.
const SMC_DEFAULT_OPTIONS = SmcColoringOptionsC(
    SMC_NONSYMMETRIC,   # structure
    SMC_COLUMN,         # partition
    SMC_DIRECT,         # decompression
    SMC_NATURAL,        # order
    Cint(0),            # postprocessing
    Cint(0),            # symmetric_pattern
    Cint(0),            # index_base (0 = C-natural)
    SMC_FLOAT64,        # dtype
)

# Out of range is an invalid argument (-3), a different failure from a
# well-formed but unsupported combination (-2).  `postprocessing` and
# `symmetric_pattern` are read as booleans, so any nonzero value is accepted.
function _valid_options(o::SmcColoringOptionsC)
    Cint(0) <= o.structure <= SMC_MAX_STRUCTURE || return false
    Cint(0) <= o.partition <= SMC_MAX_PARTITION || return false
    Cint(0) <= o.decompression <= SMC_MAX_DECOMPRESSION || return false
    Cint(0) <= o.order <= SMC_MAX_ORDER || return false
    Cint(0) <= o.dtype <= SMC_MAX_DTYPE || return false
    o.index_base == Cint(0) || o.index_base == Cint(1) || return false
    return true
end
