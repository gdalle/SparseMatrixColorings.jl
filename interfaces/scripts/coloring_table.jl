# coloring_table.jl — single source of truth for the libsmc combination tables.
#
# Pure data: no `using`, no dependency on SparseMatrixColorings, so it can be
# loaded both by `interfaces/src/c_stores.jl` (through `LibSMC.jl`) and by
# `interfaces/scripts/generate_header.jl`, into a bare `Module`.
#
# The values are part of the ABI:
# entry `i` of each enum table has the value `i - 1`; those numbers never move.

# Enumerators.  Each entry is (C enumerator, Julia
# counterpart as it appears in LibSMC.jl); its value is its 0-based position.

const DTYPES = [("SMC_FLOAT64", "Float64"), ("SMC_FLOAT32", "Float32")]

const STRUCTURES = [("SMC_NONSYMMETRIC", ":nonsymmetric"), ("SMC_SYMMETRIC", ":symmetric")]

const PARTITIONS = [
    ("SMC_COLUMN", ":column"), ("SMC_ROW", ":row"), ("SMC_BIDIRECTIONAL", ":bidirectional")
]

const DECOMPRESSIONS = [("SMC_DIRECT", ":direct"), ("SMC_SUBSTITUTION", ":substitution")]

# `RandomOrder` is excluded from v1: it carries an `AbstractRNG`, untested under
# `--trim=safe`.  Every order below is a singleton, hence a literal constructor.
const ORDERS = [
    ("SMC_NATURAL", "NaturalOrder()"),
    ("SMC_LARGEST_FIRST", "LargestFirst()"),
    ("SMC_SMALLEST_LAST", "DynamicDegreeBasedOrder{:back,:high2low,false}()"),
    ("SMC_INCIDENCE_DEGREE", "DynamicDegreeBasedOrder{:back,:low2high,false}()"),
    ("SMC_DYNAMIC_LARGEST_FIRST", "DynamicDegreeBasedOrder{:forward,:low2high,false}()"),
]

# Enum values as `Cint`: comparable with the fields of `SmcColoringOptions`.

const SMC_FLOAT64 = Cint(0)
const SMC_FLOAT32 = Cint(1)

const SMC_NONSYMMETRIC = Cint(0)
const SMC_SYMMETRIC = Cint(1)

const SMC_COLUMN = Cint(0)
const SMC_ROW = Cint(1)
const SMC_BIDIRECTIONAL = Cint(2)

const SMC_DIRECT = Cint(0)
const SMC_SUBSTITUTION = Cint(1)

const SMC_NATURAL = Cint(0)
const SMC_LARGEST_FIRST = Cint(1)
const SMC_SMALLEST_LAST = Cint(2)
const SMC_INCIDENCE_DEGREE = Cint(3)
const SMC_DYNAMIC_LARGEST_FIRST = Cint(4)

# Largest admissible value of each option, used by the -3 range checks.
const SMC_MAX_DTYPE = SMC_FLOAT32
const SMC_MAX_STRUCTURE = SMC_SYMMETRIC
const SMC_MAX_PARTITION = SMC_BIDIRECTIONAL
const SMC_MAX_DECOMPRESSION = SMC_SUBSTITUTION
const SMC_MAX_ORDER = SMC_DYNAMIC_LARGEST_FIRST

# Combo key:
#
#   key = structure * 16 + partition * 4 + decompression * 2 + dtype
#
# It fits in a `UInt8` and is what the handle -> key store records, so that
# every later call can recover the concrete type of the result it was given.

function combo_key(structure, partition, decompression, dtype)
    return UInt8(structure * 16 + partition * 4 + decompression * 2 + dtype)
end

# Key that no combination can produce: "this handle is unknown" (-> return -4).
const KEY_INVALID = 0xff

const KEY_NS_COL_DIRECT_F64 = combo_key(
    SMC_NONSYMMETRIC, SMC_COLUMN, SMC_DIRECT, SMC_FLOAT64
)
const KEY_NS_COL_DIRECT_F32 = combo_key(
    SMC_NONSYMMETRIC, SMC_COLUMN, SMC_DIRECT, SMC_FLOAT32
)
const KEY_NS_ROW_DIRECT_F64 = combo_key(SMC_NONSYMMETRIC, SMC_ROW, SMC_DIRECT, SMC_FLOAT64)
const KEY_NS_ROW_DIRECT_F32 = combo_key(SMC_NONSYMMETRIC, SMC_ROW, SMC_DIRECT, SMC_FLOAT32)
const KEY_NS_BID_DIRECT_F64 = combo_key(
    SMC_NONSYMMETRIC, SMC_BIDIRECTIONAL, SMC_DIRECT, SMC_FLOAT64
)
const KEY_NS_BID_DIRECT_F32 = combo_key(
    SMC_NONSYMMETRIC, SMC_BIDIRECTIONAL, SMC_DIRECT, SMC_FLOAT32
)
const KEY_NS_BID_SUBST_F64 = combo_key(
    SMC_NONSYMMETRIC, SMC_BIDIRECTIONAL, SMC_SUBSTITUTION, SMC_FLOAT64
)
const KEY_NS_BID_SUBST_F32 = combo_key(
    SMC_NONSYMMETRIC, SMC_BIDIRECTIONAL, SMC_SUBSTITUTION, SMC_FLOAT32
)
const KEY_SYM_COL_DIRECT_F64 = combo_key(SMC_SYMMETRIC, SMC_COLUMN, SMC_DIRECT, SMC_FLOAT64)
const KEY_SYM_COL_DIRECT_F32 = combo_key(SMC_SYMMETRIC, SMC_COLUMN, SMC_DIRECT, SMC_FLOAT32)
const KEY_SYM_COL_SUBST_F64 = combo_key(
    SMC_SYMMETRIC, SMC_COLUMN, SMC_SUBSTITUTION, SMC_FLOAT64
)
const KEY_SYM_COL_SUBST_F32 = combo_key(
    SMC_SYMMETRIC, SMC_COLUMN, SMC_SUBSTITUTION, SMC_FLOAT32
)

# The same keys with the `dtype` bit cleared: `smc_fast_coloring` never builds a
# result object and therefore never looks at `dtype`.
const BKEY_NS_COL_DIRECT = KEY_NS_COL_DIRECT_F64
const BKEY_NS_ROW_DIRECT = KEY_NS_ROW_DIRECT_F64
const BKEY_NS_BID_DIRECT = KEY_NS_BID_DIRECT_F64
const BKEY_NS_BID_SUBST = KEY_NS_BID_SUBST_F64
const BKEY_SYM_COL_DIRECT = KEY_SYM_COL_DIRECT_F64
const BKEY_SYM_COL_SUBST = KEY_SYM_COL_SUBST_F64

# The nine typed stores.  Each entry is
#   (store number, store name, [keys routed to it], structure, partition,
#    decompression, dtype or "both", concrete value type)
# for a `SparseMatrixCSC{Float64,Int64}` input.  `decompression_eltype` does not
# appear in the Column / Row / StarSet result types, so those three are
# dtype-independent: nine stores serve twelve keys.  `SubArray{...}` below is
# `SubArray{Int64,1,Vector{Int64},Tuple{UnitRange{Int64}},true}`.

const COMBOS = [
    (
        1,
        "store_ns_col_direct",
        [KEY_NS_COL_DIRECT_F64, KEY_NS_COL_DIRECT_F32],
        "SMC_NONSYMMETRIC",
        "SMC_COLUMN",
        "SMC_DIRECT",
        "both",
        "ColumnColoringResult{SparseMatrixCSC{Float64,Int64}, Int64, BipartiteGraph{Int64}, Vector{Int64}, Vector{SubArray{...}}, Vector{Int64}, Nothing}",
    ),
    (
        2,
        "store_ns_row_direct",
        [KEY_NS_ROW_DIRECT_F64, KEY_NS_ROW_DIRECT_F32],
        "SMC_NONSYMMETRIC",
        "SMC_ROW",
        "SMC_DIRECT",
        "both",
        "RowColoringResult{SparseMatrixCSC{Float64,Int64}, Int64, BipartiteGraph{Int64}, Vector{Int64}, Vector{SubArray{...}}, Vector{Int64}, Nothing}",
    ),
    (
        3,
        "store_sym_col_direct",
        [KEY_SYM_COL_DIRECT_F64, KEY_SYM_COL_DIRECT_F32],
        "SMC_SYMMETRIC",
        "SMC_COLUMN",
        "SMC_DIRECT",
        "both",
        "StarSetColoringResult{SparseMatrixCSC{Float64,Int64}, Int64, AdjacencyGraph{Int64,false}, Vector{Int64}, Vector{SubArray{...}}, Vector{Int64}, Nothing}",
    ),
    (
        4,
        "store_sym_col_subst_f64",
        [KEY_SYM_COL_SUBST_F64],
        "SMC_SYMMETRIC",
        "SMC_COLUMN",
        "SMC_SUBSTITUTION",
        "SMC_FLOAT64",
        "TreeSetColoringResult{SparseMatrixCSC{Float64,Int64}, Int64, AdjacencyGraph{Int64,false}, Vector{SubArray{...}}, Float64}",
    ),
    (
        5,
        "store_sym_col_subst_f32",
        [KEY_SYM_COL_SUBST_F32],
        "SMC_SYMMETRIC",
        "SMC_COLUMN",
        "SMC_SUBSTITUTION",
        "SMC_FLOAT32",
        "TreeSetColoringResult{SparseMatrixCSC{Float64,Int64}, Int64, AdjacencyGraph{Int64,false}, Vector{SubArray{...}}, Float32}",
    ),
    (
        6,
        "store_ns_bid_direct_f64",
        [KEY_NS_BID_DIRECT_F64],
        "SMC_NONSYMMETRIC",
        "SMC_BIDIRECTIONAL",
        "SMC_DIRECT",
        "SMC_FLOAT64",
        "BicoloringResult{SparseMatrixCSC{Float64,Int64}, Int64, AdjacencyGraph{Int64,true}, :direct, Vector{SubArray{...}}, StarSetColoringResult{SparsityPatternCSC{Int64}, Int64, AdjacencyGraph{Int64,true}, Vector{Int64}, Vector{SubArray{...}}, Vector{Int64}, Nothing}, Float64}",
    ),
    (
        7,
        "store_ns_bid_direct_f32",
        [KEY_NS_BID_DIRECT_F32],
        "SMC_NONSYMMETRIC",
        "SMC_BIDIRECTIONAL",
        "SMC_DIRECT",
        "SMC_FLOAT32",
        "BicoloringResult{SparseMatrixCSC{Float64,Int64}, Int64, AdjacencyGraph{Int64,true}, :direct, Vector{SubArray{...}}, StarSetColoringResult{SparsityPatternCSC{Int64}, Int64, AdjacencyGraph{Int64,true}, Vector{Int64}, Vector{SubArray{...}}, Vector{Int64}, Nothing}, Float32}",
    ),
    (
        8,
        "store_ns_bid_subst_f64",
        [KEY_NS_BID_SUBST_F64],
        "SMC_NONSYMMETRIC",
        "SMC_BIDIRECTIONAL",
        "SMC_SUBSTITUTION",
        "SMC_FLOAT64",
        "BicoloringResult{SparseMatrixCSC{Float64,Int64}, Int64, AdjacencyGraph{Int64,true}, :substitution, Vector{SubArray{...}}, TreeSetColoringResult{SparsityPatternCSC{Int64}, Int64, AdjacencyGraph{Int64,true}, Vector{SubArray{...}}, Float64}, Float64}",
    ),
    (
        9,
        "store_ns_bid_subst_f32",
        [KEY_NS_BID_SUBST_F32],
        "SMC_NONSYMMETRIC",
        "SMC_BIDIRECTIONAL",
        "SMC_SUBSTITUTION",
        "SMC_FLOAT32",
        "BicoloringResult{SparseMatrixCSC{Float64,Int64}, Int64, AdjacencyGraph{Int64,true}, :substitution, Vector{SubArray{...}}, TreeSetColoringResult{SparsityPatternCSC{Int64}, Int64, AdjacencyGraph{Int64,true}, Vector{SubArray{...}}, Float32}, Float32}",
    ),
]

# Every other combination is rejected with -2: `(nonsymmetric, column|row,
# substitution)` and any symmetric row or bidirectional partition are not
# `ColoringProblem`s that SparseMatrixColorings supports.
const SUPPORTED_KEYS = [k for combo in COMBOS for k in combo[3]]
