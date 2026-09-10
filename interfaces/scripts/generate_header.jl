#!/usr/bin/env julia
# Generate interfaces/include/smc.h and interfaces/include/smc.f90 from the
# function_sigs table in LibSMC.jl and the enum tables in coloring_table.jl, so
# that the C and the Fortran binding cannot drift apart.  Dependency-free: when
# either input cannot be loaded it warns and falls back to the built-in tables
# below, which must produce byte-identical output (CI regenerates and diffs).
#
# Usage:  julia interfaces/scripts/generate_header.jl

const LIBSMC_PATH = normpath(joinpath(@__DIR__, "..", "src", "LibSMC.jl"))
const COLORING_TABLE_PATH = normpath(joinpath(@__DIR__, "coloring_table.jl"))
const PROJECT_TOML_PATH = normpath(joinpath(@__DIR__, "..", "..", "Project.toml"))
const OUT_PATH = normpath(joinpath(@__DIR__, "..", "include", "smc.h"))
const OUT_PATH_F90 = normpath(joinpath(@__DIR__, "..", "include", "smc.f90"))

# Version parsed straight out of Project.toml, so the package need not load.
function project_version(path)
    for line in eachline(path)
        startswith(strip(line), '[') && break  # only the top-level table
        m = match(r"^\s*version\s*=\s*\"([^\"]+)\"", line)
        m === nothing || return VersionNumber(m.captures[1])
    end
    return error("no `version = \"...\"` entry found in $path")
end

const _SV = project_version(PROJECT_TOML_PATH)

# Enumerators: (c_type_name, [candidate names in coloring_table.jl], doc,
# [(name, value)]).  The values are the canonical ABI and must never move.
const ENUMS = [
    (
        "SmcDataType",
        [:DTYPES, :DATA_TYPES, :SMC_DTYPES],
        "Element type of the numerical buffers passed to smc_compress and\n" *
        "smc_decompress (double or float).  The sparsity pattern is always int.",
        [("SMC_FLOAT64", 0), ("SMC_FLOAT32", 1)],
    ),
    (
        "SmcStructure",
        [:STRUCTURES, :SMC_STRUCTURES],
        "Structure of the matrix.  SMC_SYMMETRIC states that the sparsity pattern\n" *
        "is symmetric and selects the symmetric coloring problems.",
        [("SMC_NONSYMMETRIC", 0), ("SMC_SYMMETRIC", 1)],
    ),
    (
        "SmcPartition",
        [:PARTITIONS, :SMC_PARTITIONS],
        "Which dimension is colored.  SMC_BIDIRECTIONAL colors rows and columns at\n" *
        "the same time and produces two compressed matrices.",
        [("SMC_COLUMN", 0), ("SMC_ROW", 1), ("SMC_BIDIRECTIONAL", 2)],
    ),
    (
        "SmcDecompression",
        [:DECOMPRESSIONS, :SMC_DECOMPRESSIONS],
        "How the nonzeros are recovered from the compressed matrix.\n" *
        "SMC_SUBSTITUTION needs fewer colors but is only available for the\n" *
        "symmetric-column and bidirectional problems.",
        [("SMC_DIRECT", 0), ("SMC_SUBSTITUTION", 1)],
    ),
    (
        "SmcOrder",
        [:ORDERS, :SMC_ORDERS],
        "Vertex order used by the greedy coloring algorithm.\n" *
        "RandomOrder is deliberately not exposed by this interface.",
        [
            ("SMC_NATURAL", 0),
            ("SMC_LARGEST_FIRST", 1),
            ("SMC_SMALLEST_LAST", 2),
            ("SMC_INCIDENCE_DEGREE", 3),
            ("SMC_DYNAMIC_LARGEST_FIRST", 4),
        ],
    ),
]

# SmcColoringOptions.  Mirrored field for field, in this exact order, by the
# isbits struct in src/c_enums.jl.
const OPTION_FIELDS = [
    ("int", "structure", "SmcStructure     - default SMC_NONSYMMETRIC"),
    ("int", "partition", "SmcPartition     - default SMC_COLUMN"),
    ("int", "decompression", "SmcDecompression - default SMC_DIRECT"),
    ("int", "order", "SmcOrder         - default SMC_NATURAL"),
    ("int", "postprocessing", "0/1 - give the neutral color 0 to the entries that need no"),
    ("int", "", "      evaluation, where possible (default 0)"),
    ("int", "symmetric_pattern", "0/1 - assert that the sparsity pattern is symmetric,"),
    ("int", "", "      skipping the symmetrization step (default 0)"),
    ("int", "index_base", "0 or 1 - index base of colptr, rowval and of the group"),
    ("int", "", "         members returned by the queries (default 0)"),
    ("int", "dtype", "SmcDataType - element type used by smc_compress and"),
    ("int", "", "              smc_decompress (default SMC_FLOAT64)"),
]

# Supported (structure, partition, decompression) combinations; anything else
# is rejected with -2.
const SUPPORTED_COMBOS = [
    ("SMC_NONSYMMETRIC", "SMC_COLUMN", "SMC_DIRECT"),
    ("SMC_NONSYMMETRIC", "SMC_ROW", "SMC_DIRECT"),
    ("SMC_SYMMETRIC", "SMC_COLUMN", "SMC_DIRECT"),
    ("SMC_SYMMETRIC", "SMC_COLUMN", "SMC_SUBSTITUTION"),
    ("SMC_NONSYMMETRIC", "SMC_BIDIRECTIONAL", "SMC_DIRECT"),
    ("SMC_NONSYMMETRIC", "SMC_BIDIRECTIONAL", "SMC_SUBSTITUTION"),
]

# Fallback prototype table, used when LibSMC.jl cannot be loaded; kept in sync
# with LibSMC.function_sigs by hand, and any disagreement is a hard error.
# Each entry: (c_name, return_type, [(arg_name, c_type), ...])
const FALLBACK_FUNCTION_SIGS = Tuple{String,String,Vector{Tuple{String,String}}}[
    ("smc_default_options", "SmcColoringOptions", []),
    ("smc_version", "void", [("major", "int*"), ("minor", "int*"), ("patch", "int*")]),
    (
        "smc_coloring",
        "int",
        [
            ("m", "int"),
            ("n", "int"),
            ("colptr", "const int*"),
            ("rowval", "const int*"),
            ("opts", "const SmcColoringOptions*"),
            ("result_out", "void**"),
        ],
    ),
    (
        "smc_fast_coloring",
        "int",
        [
            ("m", "int"),
            ("n", "int"),
            ("colptr", "const int*"),
            ("rowval", "const int*"),
            ("opts", "const SmcColoringOptions*"),
            ("row_colors", "int*"),
            ("column_colors", "int*"),
            ("ncolors_out", "int*"),
        ],
    ),
    ("smc_result_free", "int", [("result", "void*")]),
    ("smc_ncolors", "int", [("result", "void*"), ("ncolors_out", "int*")]),
    ("smc_column_colors", "int", [("result", "void*"), ("colors", "int*"), ("len", "int")]),
    ("smc_row_colors", "int", [("result", "void*"), ("colors", "int*"), ("len", "int")]),
    ("smc_ncolumn_groups", "int", [("result", "void*"), ("ngroups_out", "int*")]),
    ("smc_nrow_groups", "int", [("result", "void*"), ("ngroups_out", "int*")]),
    (
        "smc_column_group_size",
        "int",
        [("result", "void*"), ("group", "int"), ("size_out", "int*")],
    ),
    (
        "smc_column_group",
        "int",
        [("result", "void*"), ("group", "int"), ("members", "int*"), ("len", "int")],
    ),
    (
        "smc_row_group_size",
        "int",
        [("result", "void*"), ("group", "int"), ("size_out", "int*")],
    ),
    (
        "smc_row_group",
        "int",
        [("result", "void*"), ("group", "int"), ("members", "int*"), ("len", "int")],
    ),
    ("smc_nnz", "int", [("result", "void*"), ("nnz_out", "int*")]),
    ("smc_size", "int", [("result", "void*"), ("m_out", "int*"), ("n_out", "int*")]),
    (
        "smc_compressed_size",
        "int",
        [
            ("result", "void*"),
            ("Br_rows", "int*"),
            ("Br_cols", "int*"),
            ("Bc_rows", "int*"),
            ("Bc_cols", "int*"),
        ],
    ),
    (
        "smc_compress",
        "int",
        [
            ("result", "void*"),
            ("nzval", "const void*"),
            ("nzval_len", "size_t"),
            ("Br", "void*"),
            ("Br_len", "size_t"),
            ("Bc", "void*"),
            ("Bc_len", "size_t"),
        ],
    ),
    (
        "smc_decompress",
        "int",
        [
            ("result", "void*"),
            ("Br", "const void*"),
            ("Br_len", "size_t"),
            ("Bc", "const void*"),
            ("Bc_len", "size_t"),
            ("A_out", "void*"),
            ("A_len", "size_t"),
        ],
    ),
]

# Per-function documentation, emitted as a C comment before each prototype.
const FUNCTION_DOCS = Dict{String,String}(
    "smc_default_options" =>
        "Return an SmcColoringOptions filled with the defaults: nonsymmetric\n" *
        "structure, column partition, direct decompression, natural order, no\n" *
        "postprocessing, no symmetric-pattern assertion, 0-based indices and\n" *
        "SMC_FLOAT64.  Always initialise an options struct with this call before\n" *
        "overriding individual fields.",
    "smc_version" =>
        "Write the SparseMatrixColorings.jl version of this library into\n" *
        "*major, *minor, *patch (the same values as the SMC_VERSION_* macros).",
    "smc_coloring" =>
        "Color the m-by-n sparsity pattern given in CSC form and return an opaque\n" *
        "result handle through *result_out.  Only the structure is needed here;\n" *
        "the numerical values are passed later to smc_compress.\n" *
        "  m, n       : number of rows and columns, both > 0\n" *
        "  colptr     : n+1 column pointers, in opts->index_base\n" *
        "  rowval     : row indices of the nonzeros, in opts->index_base,\n" *
        "               length colptr[n] - colptr[0]\n" *
        "  opts       : coloring options, or NULL for smc_default_options()\n" *
        "  result_out : receives the handle; release it with smc_result_free\n" *
        "Returns 0, -1 on an internal error, -2 if the combination of structure,\n" *
        "partition, decompression and dtype is unsupported, -3 on an invalid\n" *
        "argument.",
    "smc_fast_coloring" =>
        "Color the pattern and write the colors directly, without allocating a\n" *
        "handle.  Convenient when only the colors are needed; the groups and the\n" *
        "compression helpers require smc_coloring instead.\n" *
        "  row_colors    : length-m buffer, may be NULL when the partition\n" *
        "                  produces no row coloring (SMC_COLUMN)\n" *
        "  column_colors : length-n buffer, may be NULL when the partition\n" *
        "                  produces no column coloring (SMC_ROW)\n" *
        "  ncolors_out   : receives the number of colors\n" *
        "SMC_BIDIRECTIONAL fills both buffers, so neither may be NULL.\n" *
        "Colors are labels in 1..ncolors; 0 marks an entry that needs no\n" *
        "evaluation and can only appear when opts->postprocessing is 1.  Color\n" *
        "labels are never shifted by opts->index_base.\n" *
        "Returns 0, -1 on an internal error, -2 on an unsupported combination,\n" *
        "-3 on an invalid argument.",
    "smc_result_free" =>
        "Release a handle returned by smc_coloring; it must not be used again.\n" *
        "Returns 0, or -4 if the handle is unknown (freeing twice is safe).",
    "smc_ncolors" =>
        "Write the total number of colors of the result into *ncolors_out.\n" *
        "Returns 0, -3 on an invalid argument, -4 on an invalid handle.",
    "smc_column_colors" =>
        "Copy the color of every column into `colors`; `len` must be at least n.\n" *
        "Colors are labels in 1..ncolors, 0 meaning \"no evaluation needed\".\n" *
        "Returns 0, -2 if the partition has no column coloring, -3 on an invalid\n" *
        "argument (including len < n), -4 on an invalid handle.",
    "smc_row_colors" =>
        "Copy the color of every row into `colors`; `len` must be at least m.\n" *
        "Colors are labels in 1..ncolors, 0 meaning \"no evaluation needed\".\n" *
        "Returns 0, -2 if the partition has no row coloring, -3 on an invalid\n" *
        "argument (including len < m), -4 on an invalid handle.",
    "smc_ncolumn_groups" =>
        "Write the number of column groups into *ngroups_out.  Groups are the\n" *
        "color classes: group g holds every column colored g.\n" *
        "Returns 0, -2 if the partition has no column coloring, -3 on an invalid\n" *
        "argument, -4 on an invalid handle.",
    "smc_nrow_groups" =>
        "Write the number of row groups into *ngroups_out.\n" *
        "Returns 0, -2 if the partition has no row coloring, -3 on an invalid\n" *
        "argument, -4 on an invalid handle.",
    "smc_column_group_size" =>
        "Write the number of columns in column group `group` into *size_out.\n" *
        "`group` is 1-based and runs over 1..smc_ncolumn_groups, independently of\n" *
        "opts->index_base.  Query the size first, then fetch the members.\n" *
        "Returns 0, -2 if the partition has no column coloring, -3 on an invalid\n" *
        "argument (including an out-of-range group), -4 on an invalid handle.",
    "smc_column_group" =>
        "Copy the column indices of column group `group` into `members`; `len`\n" *
        "must be at least smc_column_group_size(result, group).  The indices are\n" *
        "written in opts->index_base.\n" *
        "Returns 0, -2 if the partition has no column coloring, -3 on an invalid\n" *
        "argument (including len too small), -4 on an invalid handle.",
    "smc_row_group_size" =>
        "Write the number of rows in row group `group` into *size_out.  `group`\n" *
        "is 1-based and runs over 1..smc_nrow_groups.\n" *
        "Returns 0, -2 if the partition has no row coloring, -3 on an invalid\n" *
        "argument (including an out-of-range group), -4 on an invalid handle.",
    "smc_row_group" =>
        "Copy the row indices of row group `group` into `members`; `len` must be\n" *
        "at least smc_row_group_size(result, group).  The indices are written in\n" *
        "opts->index_base.\n" *
        "Returns 0, -2 if the partition has no row coloring, -3 on an invalid\n" *
        "argument (including len too small), -4 on an invalid handle.",
    "smc_nnz" =>
        "Write the number of stored entries of the sparsity pattern this result\n" *
        "was built from into *nnz_out.  That is exactly the number of elements\n" *
        "`nzval` must have in smc_compress, i.e. the required nzval_len.\n" *
        "Returns 0, -3 on an invalid argument, -4 on an invalid handle.",
    "smc_size" =>
        "Write the dimensions of the matrix this result was built from into\n" *
        "*m_out and *n_out.  They are the lengths expected by smc_row_colors (m)\n" *
        "and smc_column_colors (n), and A_out in smc_decompress must hold m*n\n" *
        "elements, i.e. A_len must be at least m*n.\n" *
        "Both out pointers must be non-NULL.\n" *
        "Returns 0, -3 on an invalid argument, -4 on an invalid handle.",
    "smc_compressed_size" =>
        "Report the dimensions of the compressed matrices, so the caller can size\n" *
        "the buffers of smc_compress and smc_decompress.\n" *
        "  Bc : m-by-ncolors for a column partition, ncolors-by-n for a row\n" *
        "       partition, m-by-ncolumn_groups for a bidirectional one\n" *
        "  Br : nrow_groups-by-n, and used only by SMC_BIDIRECTIONAL; for the\n" *
        "       other partitions *Br_rows and *Br_cols are set to 0\n" *
        "The Bc_len and Br_len arguments of smc_compress and smc_decompress must\n" *
        "be at least Bc_rows*Bc_cols and Br_rows*Br_cols respectively.\n" *
        "All four out pointers must be non-NULL.\n" *
        "Returns 0, -3 on an invalid argument, -4 on an invalid handle.",
    "smc_compress" =>
        "Compress the matrix into the dense buffers Br and Bc.  Every buffer is\n" *
        "followed by its length, counted in elements of the type selected by\n" *
        "opts->dtype -- never in bytes.\n" *
        "  nzval, nzval_len : the CSC values, in the same order as the rowval\n" *
        "                     given to smc_coloring; double* or float* according\n" *
        "                     to opts->dtype.  nzval_len must be at least the\n" *
        "                     value reported by smc_nnz\n" *
        "  Br, Br_len       : row-compressed matrix, used only by\n" *
        "                     SMC_BIDIRECTIONAL; for the other partitions Br\n" *
        "                     may be NULL and Br_len 0.  A bidirectional result\n" *
        "                     requires it, with Br_len at least Br_rows*Br_cols\n" *
        "                     from smc_compressed_size\n" *
        "  Bc, Bc_len       : column-compressed matrix; Bc_len must be at least\n" *
        "                     Bc_rows*Bc_cols from smc_compressed_size\n" *
        "Both buffers are column-major with the dimensions reported by\n" *
        "smc_compressed_size: B[i,j] is B[j*rows + i].\n" *
        "Returns 0, -1 on an internal error, -3 on an invalid argument (a NULL\n" *
        "buffer the partition needs, or a buffer too small), -4 on an invalid\n" *
        "handle.",
    "smc_decompress" =>
        "Recover the full m-by-n dense matrix from the compressed form.  Every\n" *
        "buffer is followed by its length, counted in elements of the type\n" *
        "selected by opts->dtype -- never in bytes.\n" *
        "  Br, Br_len   : the buffer filled by smc_compress, used only by\n" *
        "                 SMC_BIDIRECTIONAL; for the other partitions Br may be\n" *
        "                 NULL and Br_len 0.  A bidirectional result requires\n" *
        "                 it, with Br_len at least Br_rows*Br_cols from\n" *
        "                 smc_compressed_size\n" *
        "  Bc, Bc_len   : the buffer filled by smc_compress; Bc_len must be at\n" *
        "                 least Bc_rows*Bc_cols from smc_compressed_size\n" *
        "  A_out, A_len : m*n elements, column-major, of the type selected by\n" *
        "                 opts->dtype; A_out[i,j] is A_out[j*m + i].  A_len must\n" *
        "                 be at least m*n, with m and n from smc_size\n" *
        "Entries outside the sparsity pattern are set to zero.\n" *
        "Returns 0, -1 on an internal error, -3 on an invalid argument (a NULL\n" *
        "buffer the partition needs, or a buffer too small), -4 on an invalid\n" *
        "handle.",
)

# Section banners, emitted just before the named function.
const SECTIONS = [
    (
        "smc_coloring",
        [
            "Coloring",
            "",
            "The pattern is always given in CSC form, in opts->index_base.  The",
            "caller's arrays are copied and never modified.",
        ],
    ),
    (
        "smc_ncolors",
        [
            "Queries",
            "",
            "All of them take a handle from smc_coloring.  Every buffer crossing",
            "the interface carries its own length, that length is checked before a",
            "single element is read or written, and every sizing question has a",
            "query, so a caller can always ask before allocating:",
            "",
            "  buffer   length argument   how to obtain the required length",
            "  colors   len               n (columns) or m (rows), from smc_size",
            "  members  len               smc_column_group_size / smc_row_group_size",
            "  nzval    nzval_len         smc_nnz",
            "  Bc       Bc_len            Bc_rows*Bc_cols, smc_compressed_size",
            "  Br       Br_len            Br_rows*Br_cols, smc_compressed_size",
            "  A_out    A_len             m*n, from smc_size",
            "",
            "Lengths are element counts, never byte counts.  The numerical",
            "buffers use size_t rather than int because A_len is m*n, which",
            "overflows a 32-bit int.  Too small a buffer is rejected with -3.",
        ],
    ),
    (
        "smc_compressed_size",
        [
            "Compression / decompression",
            "",
            "Dense matrices are column-major (Fortran / Julia order) and hold",
            "double or float elements according to opts->dtype.",
        ],
    ),
]

# Optional inputs: LibSMC.function_sigs and the enum tables of coloring_table.jl
function load_function_sigs()
    isfile(LIBSMC_PATH) || return nothing
    try
        Base.include(Main, LIBSMC_PATH)
        # invokelatest: the bindings only exist in a newer world than this function.
        lib = Base.invokelatest(getglobal, Main, :LibSMC)
        sigs = Base.invokelatest(getglobal, lib, :function_sigs)
        return Tuple{String,String,Vector{Tuple{String,String}}}[
            (
                String(name),
                String(ret),
                Tuple{String,String}[(String(a), String(t)) for (a, t) in args],
            ) for (name, ret, args) in sigs
        ]
    catch e
        @warn "could not read function_sigs from $LIBSMC_PATH; using the built-in table" exception =
            e
        return nothing
    end
end

function load_coloring_table()
    isfile(COLORING_TABLE_PATH) || return nothing
    try
        mod = Module(:SmcColoringTable)
        Base.include(mod, COLORING_TABLE_PATH)
        return mod
    catch e
        @warn "could not load $COLORING_TABLE_PATH; using the built-in enum values" exception =
            e
        return nothing
    end
end

# Pull a C enumerator name out of a table entry, whatever its shape (bare
# string, or a tuple / pair / vector holding one somewhere).
function enumerator_name(entry)
    entry isa AbstractString && return startswith(entry, "SMC_") ? String(entry) : nothing
    parts = if entry isa Pair
        (entry.first, entry.second)
    elseif entry isa Union{Tuple,AbstractVector}
        entry
    else
        ()
    end
    for x in parts
        x isa AbstractString && startswith(x, "SMC_") && return String(x)
    end
    return nothing
end

# Enumerators of `enum_name`, as declared by coloring_table.jl, or nothing.
function enumerators_from_table(mod, candidates)
    mod === nothing && return nothing
    for sym in candidates
        isdefined(mod, sym) || continue
        tbl = Base.invokelatest(getglobal, mod, sym)
        tbl isa AbstractVector && !isempty(tbl) || continue
        found = String[]
        for entry in tbl
            name = enumerator_name(entry)
            if name === nothing
                empty!(found)
                break
            end
            push!(found, name)
        end
        isempty(found) || return [(name, i - 1) for (i, name) in enumerate(found)]
    end
    return nothing
end

const coloring_table = load_coloring_table()
const loaded_sigs = load_function_sigs()

# Canonical values stay authoritative: the header is the ABI, so a disagreement
# is reported instead of silently changing the generated file.
function checked_enumerators(type_name, candidates, canonical)
    from_table = enumerators_from_table(coloring_table, candidates)
    if from_table !== nothing && from_table != canonical
        @warn "coloring_table.jl disagrees with the built-in values for $type_name; keeping the built-in ones" builtin =
            canonical table = from_table
    end
    return canonical
end

function checked_sigs()
    loaded_sigs === nothing && return FALLBACK_FUNCTION_SIGS
    if loaded_sigs != FALLBACK_FUNCTION_SIGS
        # Diff the FULL signatures, not just the names: a name-only diff lets an
        # arity or type change through, and the header then misdeclares the ABI to
        # every C caller.  Nothing else in the test suite or in CI catches that.
        render(sig) = string(
            sig[2],
            " ",
            sig[1],
            "(",
            join([string(t, " ", n) for (n, t) in sig[3]], ", "),
            ")",
        )
        by_name = Dict(s[1] => s for s in FALLBACK_FUNCTION_SIGS)
        problems = String[]
        for name in setdiff(first.(FALLBACK_FUNCTION_SIGS), first.(loaded_sigs))
            push!(
                problems, "  $name: in the built-in table but not in LibSMC.function_sigs"
            )
        end
        for sig in loaded_sigs
            reference = get(by_name, sig[1], nothing)
            if reference === nothing
                push!(
                    problems,
                    "  $(sig[1]): in LibSMC.function_sigs but not in the built-in table",
                )
            elseif sig != reference
                push!(problems, "  $(sig[1]):")
                push!(problems, "      LibSMC:   " * render(sig))
                push!(problems, "      built-in: " * render(reference))
            end
        end
        error(
            "LibSMC.function_sigs disagrees with the built-in table in generate_header.jl.\n" *
            "Both describe the C ABI, so a disagreement means the header would misdeclare it.\n" *
            "Reconcile them (they must match element for element), then re-run:\n" *
            join(problems, "\n"),
        )
    end
    return loaded_sigs
end

# The one checked ABI description, shared by the C and the Fortran emitters.
const SIGS = checked_sigs()
const ENUM_TABLES = [
    (type_name, doc, checked_enumerators(type_name, candidates, canonical)) for
    (type_name, candidates, doc, canonical) in ENUMS
]

# Emitters.

# A C comment block: single line as /* ... */, multi-line as a /* * */ block.
function emit_doc(io, text)
    lines = split(text, '\n')
    if length(lines) == 1
        println(io, "/* $(lines[1]) */")
    else
        println(io, "/*")
        for l in lines
            println(io, isempty(l) ? " *" : " * $l")
        end
        println(io, " */")
    end
end

function emit_banner(io, lines)
    println(
        io, "/* -------------------------------------------------------------------------"
    )
    for l in lines
        println(io, isempty(l) ? " *" : " * $l")
    end
    return println(
        io,
        " * ------------------------------------------------------------------------- */",
    )
end

function emit_enum(io, type_name, doc, entries)
    emit_doc(io, doc)
    println(io, "typedef enum {")
    width = maximum(length(name) for (name, _) in entries)
    for (i, (name, value)) in enumerate(entries)
        comma = i == length(entries) ? "" : ","
        println(io, "  $(rpad(name, width)) = $value$comma")
    end
    println(io, "} $type_name;")
    return println(io)
end

function emit_options_struct(io)
    println(io, "typedef struct {")
    decls = [isempty(name) ? "" : "$ctype $name;" for (ctype, name, _) in OPTION_FIELDS]
    dwidth = maximum(length, decls)
    cwidth = maximum(length(comment) for (_, _, comment) in OPTION_FIELDS)
    for ((_, _, comment), decl) in zip(OPTION_FIELDS, decls)
        println(io, "  $(rpad(decl, dwidth))  /* $(rpad(comment, cwidth)) */")
    end
    println(io, "} SmcColoringOptions;")
    return println(io)
end

# Write smc.h
mkpath(dirname(OUT_PATH))

open(OUT_PATH, "w") do io
    println(
        io,
        """
#ifndef SMC_H
#define SMC_H

#include <stddef.h> /* size_t */

/* Version */
#define SMC_VERSION_MAJOR $(_SV.major)
#define SMC_VERSION_MINOR $(_SV.minor)
#define SMC_VERSION_PATCH $(_SV.patch)

#ifdef __cplusplus
extern "C" {
#endif
""",
    )

    emit_banner(
        io,
        [
            "libsmc - C interface to SparseMatrixColorings.jl",
            "",
            "Typical use:",
            "",
            "  SmcColoringOptions opts = smc_default_options();",
            "  opts.structure = SMC_NONSYMMETRIC;",
            "  opts.partition = SMC_COLUMN;",
            "  opts.order     = SMC_LARGEST_FIRST;",
            "",
            "  void *result;",
            "  if (smc_coloring(m, n, colptr, rowval, &opts, &result) != 0) { ... }",
            "",
            "  int nc;",
            "  smc_ncolors(result, &nc);",
            "  int *colors = malloc(n * sizeof(int));",
            "  smc_column_colors(result, colors, n);",
            "",
            "  int Br_rows, Br_cols, Bc_rows, Bc_cols;",
            "  smc_compressed_size(result, &Br_rows, &Br_cols, &Bc_rows, &Bc_cols);",
            "  size_t Bc_len = (size_t) Bc_rows * (size_t) Bc_cols;",
            "  double *Bc = malloc(Bc_len * sizeof(double));",
            "",
            "  int nnz;",
            "  smc_nnz(result, &nnz);",
            "  smc_compress(result, nzval, (size_t) nnz, NULL, 0, Bc, Bc_len);",
            "",
            "  smc_result_free(result);",
            "",
            "The sparsity pattern is passed in compressed sparse column form: colptr",
            "has n+1 entries, rowval has colptr[n] - colptr[0] entries, and both use",
            "the index base selected by opts.index_base (0 by default).  The caller's",
            "arrays are copied, never modified.  Coloring is structure-only: the",
            "numerical values are needed only by smc_compress.",
            "",
            "Indices crossing the interface are 32-bit int; matrices with more than",
            "2^31 nonzeros are out of scope.  Dense matrices are column-major, with",
            "double or float elements according to opts.dtype.",
            "",
            "Every output buffer is caller-allocated and every buffer argument is",
            "immediately followed by its length, counted in elements, never in bytes.",
        ],
    )
    println(io)

    emit_banner(io, ["Enumerators"])
    println(io)
    for (type_name, doc, entries) in ENUM_TABLES
        emit_enum(io, type_name, doc, entries)
    end

    combo_lines = ["  $(rpad(s, 16)) $(rpad(p, 17)) $d" for (s, p, d) in SUPPORTED_COMBOS]
    emit_banner(
        io,
        [
            "Coloring options",
            "",
            "Passed to smc_coloring and smc_fast_coloring, and remembered by the",
            "result handle.  Initialise with smc_default_options() before overriding",
            "individual fields; a NULL options pointer means the defaults.",
            "",
            "Supported (structure, partition, decompression) combinations; anything",
            "else is rejected with -2:",
            combo_lines...,
        ],
    )
    println(io)
    emit_options_struct(io)

    emit_banner(
        io,
        [
            "Return codes",
            "",
            "Every function returning int returns one of:",
            "",
            "   0  success",
            "  -1  internal error (a Julia exception was caught and logged)",
            "  -2  unsupported combination of (structure, partition, decompression,",
            "      dtype)",
            "  -3  invalid argument (NULL pointer, bad dimension, buffer too small,",
            "      bad enum value, bad index_base)",
            "  -4  invalid or already-freed handle",
        ],
    )
    println(io)

    emit_banner(io, ["API functions"])
    println(io)

    sections = Dict(SECTIONS)
    for (name, ret, args) in SIGS
        if haskey(sections, name)
            emit_banner(io, sections[name])
            println(io)
        end
        if haskey(FUNCTION_DOCS, name)
            emit_doc(io, FUNCTION_DOCS[name])
        else
            @warn "no documentation for $name in FUNCTION_DOCS; emitting a bare prototype"
        end
        arg_str = if isempty(args)
            "void"
        else
            join(["$ctype $aname" for (aname, ctype) in args], ", ")
        end
        println(io, "$ret $name($arg_str);")
        println(io)
    end

    return println(
        io,
        """
#ifdef __cplusplus
}
#endif

#endif /* SMC_H */""",
    )
end

println("Generated $OUT_PATH")

# ===========================================================================
# Fortran binding (include/smc.f90), emitted from SIGS and ENUM_TABLES like
# smc.h above.  Following Krylov.jl it is an INCLUDE FILE rather than a module
# (`use iso_c_binding`, `implicit none`, then `include 'smc.f90'`), which
# avoids shipping compiler-specific .mod files.
# ===========================================================================

# C-to-Fortran type mapping.  Every C pointer becomes a `type(c_ptr), value`
# that the caller fills with c_loc(x) or c_null_ptr; the only exception is the
# `void**` out-parameter, which becomes an `intent(out)` c_ptr.
#
# `smc_version`'s three `int*` outputs are instead plain `integer(c_int),
# intent(out)` scalars, so a caller writes `call smc_version(major, minor,
# patch)`; Fortran passes scalars by reference, so this is the same ABI.  It is
# a per-argument override rather than a blanket `int* -> intent(out)` rule
# because `int*` is also used for output *arrays* (`colors`, `members`), which
# must stay `type(c_ptr)` so the caller can pass c_loc of an array section.
const FORTRAN_ARG_OVERRIDES = Dict{Tuple{String,String},Tuple{String,String}}(
    ("smc_version", "major") => ("integer(c_int)", "intent(out)"),
    ("smc_version", "minor") => ("integer(c_int)", "intent(out)"),
    ("smc_version", "patch") => ("integer(c_int)", "intent(out)"),
)

function fortran_arg_type(ctype, fname="", argname="")
    override = get(FORTRAN_ARG_OVERRIDES, (fname, argname), nothing)
    override === nothing || return override
    ctype == "int" && return ("integer(c_int)", "value")
    ctype == "size_t" && return ("integer(c_size_t)", "value")
    endswith(ctype, "**") && return ("type(c_ptr)", "intent(out)")
    endswith(ctype, "*") && return ("type(c_ptr)", "value")
    return error("no Fortran mapping for the C argument type `$ctype`")
end

# Return type: nothing for `void` (a subroutine), else (ftype, result name).
function fortran_return_type(ctype)
    ctype == "void" && return nothing
    ctype == "int" && return ("integer(c_int)", "ret")
    ctype == "SmcColoringOptions" && return ("type(SmcColoringOptions)", "opts")
    return error("no Fortran mapping for the C return type `$ctype`")
end

# Per-function documentation for the Fortran binding: shorter and
# Fortran-flavoured (c_loc / c_null_ptr / opts%field) than FUNCTION_DOCS.
const FORTRAN_DOCS = Dict{String,String}(
    "smc_default_options" =>
        "Return an SmcColoringOptions filled with the defaults: nonsymmetric\n" *
        "structure, column partition, direct decompression, natural order, no\n" *
        "postprocessing, no symmetric-pattern assertion, 0-based indices and\n" *
        "SMC_FLOAT64.  Always start from this call, override the fields you\n" *
        "need, then pass c_loc(opts).",
    "smc_version" =>
        "Write the SparseMatrixColorings.jl version of this library into major,\n" *
        "minor and patch (the same values as the SMC_VERSION_* parameters).\n" *
        "Pass c_loc of three integer(c_int), target scalars.",
    "smc_coloring" =>
        "Color the m-by-n sparsity pattern given in CSC form and return an\n" *
        "opaque result handle through result_out.\n" *
        "  m, n       : number of rows and columns, both > 0\n" *
        "  colptr     : c_loc of n+1 column pointers, in opts%index_base\n" *
        "  rowval     : c_loc of the row indices of the nonzeros, in\n" *
        "               opts%index_base, length colptr(n+1) - colptr(1)\n" *
        "  opts       : c_loc(options), or c_null_ptr for the defaults\n" *
        "  result_out : receives the handle; release it with smc_result_free\n" *
        "Returns 0, -1 internal error, -2 unsupported combination, -3 invalid\n" *
        "argument.",
    "smc_fast_coloring" =>
        "Color the pattern and write the colors directly, without allocating a\n" *
        "handle.  The groups and the compression helpers need smc_coloring.\n" *
        "  row_colors    : c_loc of a length-m buffer, c_null_ptr when the\n" *
        "                  partition produces no row coloring (SMC_COLUMN)\n" *
        "  column_colors : c_loc of a length-n buffer, c_null_ptr when the\n" *
        "                  partition produces no column coloring (SMC_ROW)\n" *
        "  ncolors_out   : c_loc of a scalar receiving the number of colors\n" *
        "SMC_BIDIRECTIONAL fills both buffers, so neither may be c_null_ptr.\n" *
        "Colors are labels in 1..ncolors; 0 marks an entry that needs no\n" *
        "evaluation and can only appear when opts%postprocessing is 1.\n" *
        "Returns 0, -1 internal error, -2 unsupported combination, -3 invalid\n" *
        "argument.",
    "smc_result_free" =>
        "Release a handle returned by smc_coloring; it must not be used again.\n" *
        "Returns 0, or -4 if the handle is unknown (freeing twice is safe).",
    "smc_ncolors" =>
        "Write the total number of colors into the scalar pointed to by\n" *
        "ncolors_out.  Returns 0, -3 invalid argument, -4 invalid handle.",
    "smc_column_colors" =>
        "Copy the color of every column into colors; len must be at least n.\n" *
        "Colors are labels in 1..ncolors, 0 meaning \"no evaluation needed\".\n" *
        "Returns 0, -2 if the partition has no column coloring, -3 invalid\n" *
        "argument (including len < n), -4 invalid handle.",
    "smc_row_colors" =>
        "Copy the color of every row into colors; len must be at least m.\n" *
        "Colors are labels in 1..ncolors, 0 meaning \"no evaluation needed\".\n" *
        "Returns 0, -2 if the partition has no row coloring, -3 invalid\n" *
        "argument (including len < m), -4 invalid handle.",
    "smc_ncolumn_groups" =>
        "Write the number of column groups into ngroups_out.  Groups are the\n" *
        "color classes: group g holds every column colored g.\n" *
        "Returns 0, -2 if the partition has no column coloring, -3 invalid\n" *
        "argument, -4 invalid handle.",
    "smc_nrow_groups" =>
        "Write the number of row groups into ngroups_out.\n" *
        "Returns 0, -2 if the partition has no row coloring, -3 invalid\n" *
        "argument, -4 invalid handle.",
    "smc_column_group_size" =>
        "Write the number of columns in column group `group` into size_out.\n" *
        "`group` is 1-based and runs over 1..smc_ncolumn_groups, independently\n" *
        "of opts%index_base.  Query the size first, then fetch the members.\n" *
        "Returns 0, -2 if the partition has no column coloring, -3 invalid\n" *
        "argument (including an out-of-range group), -4 invalid handle.",
    "smc_column_group" =>
        "Copy the column indices of column group `group` into members; len must\n" *
        "be at least smc_column_group_size(result, group, ...).  The indices are\n" *
        "written in opts%index_base.\n" *
        "Returns 0, -2 if the partition has no column coloring, -3 invalid\n" *
        "argument (including len too small), -4 invalid handle.",
    "smc_row_group_size" =>
        "Write the number of rows in row group `group` into size_out.  `group`\n" *
        "is 1-based and runs over 1..smc_nrow_groups.\n" *
        "Returns 0, -2 if the partition has no row coloring, -3 invalid\n" *
        "argument (including an out-of-range group), -4 invalid handle.",
    "smc_row_group" =>
        "Copy the row indices of row group `group` into members; len must be at\n" *
        "least smc_row_group_size(result, group, ...).  The indices are written\n" *
        "in opts%index_base.\n" *
        "Returns 0, -2 if the partition has no row coloring, -3 invalid\n" *
        "argument (including len too small), -4 invalid handle.",
    "smc_nnz" =>
        "Write the number of stored entries of the sparsity pattern this result\n" *
        "was built from into nnz_out.  That is exactly the required nzval_len\n" *
        "of smc_compress.\n" *
        "Returns 0, -3 invalid argument, -4 invalid handle.",
    "smc_size" =>
        "Write the dimensions of the matrix this result was built from into\n" *
        "m_out and n_out: the lengths expected by smc_row_colors (m) and\n" *
        "smc_column_colors (n), and A_len must be at least m*n.\n" *
        "Both pointers must be non-NULL.\n" *
        "Returns 0, -3 invalid argument, -4 invalid handle.",
    "smc_compressed_size" =>
        "Report the dimensions of the compressed matrices, so the caller can\n" *
        "size the buffers of smc_compress and smc_decompress.\n" *
        "  Bc : m-by-ncolors for a column partition, ncolors-by-n for a row\n" *
        "       partition, m-by-ncolumn_groups for a bidirectional one\n" *
        "  Br : nrow_groups-by-n, used only by SMC_BIDIRECTIONAL; for the other\n" *
        "       partitions Br_rows and Br_cols are set to 0\n" *
        "All four pointers must be non-NULL.\n" *
        "Returns 0, -3 invalid argument, -4 invalid handle.",
    "smc_compress" =>
        "Compress the matrix into the dense buffers Br and Bc.  Every buffer is\n" *
        "followed by its length, counted in elements of the type selected by\n" *
        "opts%dtype -- never in bytes.\n" *
        "  nzval, nzval_len : c_loc of the CSC values, in the same order as the\n" *
        "                     rowval given to smc_coloring, real(c_double) or\n" *
        "                     real(c_float) according to opts%dtype; nzval_len\n" *
        "                     at least smc_nnz\n" *
        "  Br, Br_len       : row-compressed matrix, used only by\n" *
        "                     SMC_BIDIRECTIONAL; otherwise c_null_ptr and 0\n" *
        "  Bc, Bc_len       : column-compressed matrix, at least Bc_rows*Bc_cols\n" *
        "Both buffers are column-major with the dimensions reported by\n" *
        "smc_compressed_size, which is Fortran's own layout: B(i,j).\n" *
        "Returns 0, -1 internal error, -3 invalid argument (a c_null_ptr the\n" *
        "partition needs, or a buffer too small), -4 invalid handle.",
    "smc_decompress" =>
        "Recover the full m-by-n dense matrix from the compressed form.  Every\n" *
        "buffer is followed by its length, counted in elements of the type\n" *
        "selected by opts%dtype -- never in bytes.\n" *
        "  Br, Br_len   : the buffer filled by smc_compress, used only by\n" *
        "                 SMC_BIDIRECTIONAL; otherwise c_null_ptr and 0\n" *
        "  Bc, Bc_len   : the buffer filled by smc_compress, at least\n" *
        "                 Bc_rows*Bc_cols from smc_compressed_size\n" *
        "  A_out, A_len : m*n elements, column-major, of the type selected by\n" *
        "                 opts%dtype; A_len at least m*n, with m and n from\n" *
        "                 smc_size\n" *
        "Entries outside the sparsity pattern are set to zero.\n" *
        "Returns 0, -1 internal error, -3 invalid argument (a c_null_ptr the\n" *
        "partition needs, or a buffer too small), -4 invalid handle.",
)

const F_RULE = "-------------------------------------------------------------------------"

# Section banners are shared with the C header, which writes `opts->dtype`;
# rewriting to `opts%dtype` avoids a second copy of the prose that could drift.
f_prose(line) = replace(line, "opts->" => "opts%")

f_comment(indent, line) = isempty(line) ? "$(indent)!" : "$(indent)! $(f_prose(line))"

function emit_f_doc(io, indent, text)
    for l in split(text, '\n')
        println(io, f_comment(indent, l))
    end
end

function emit_f_banner(io, indent, lines)
    println(io, f_comment(indent, F_RULE))
    for l in lines
        println(io, f_comment(indent, l))
    end
    return println(io, f_comment(indent, F_RULE))
end

# Wrap `head(a, b, c)` over several `&`-continued lines of at most `width`
# characters, aligned under the opening parenthesis as in krylov.f90.
function wrapped_call(head, argnames, tail; width=92)
    open_col = length(head) + 1
    isempty(argnames) && return [head * "()" * tail]
    lines = String[]
    current = head * "("
    for (i, a) in enumerate(argnames)
        piece = i == length(argnames) ? a : a * ", "
        # + 2 for the trailing " &" that a continued line needs
        if length(current) + length(piece) + 2 > width && current != head * "("
            push!(lines, current * "&")
            current = " "^open_col
        end
        current *= piece
    end
    push!(lines, current * ")" * tail)
    return lines
end

# One interface body: `function`/`subroutine` statement, dummy argument
# declarations, result declaration, closing `end`.
function emit_f_interface(io, name, ret, args)
    indent = "    "
    body = "      "
    retinfo = fortran_return_type(ret)
    kind = retinfo === nothing ? "subroutine" : "function"

    # Group consecutive arguments sharing a Fortran type and attribute.
    groups = Tuple{String,String,Vector{String}}[]
    for (aname, ctype) in args
        ftype, attr = fortran_arg_type(ctype, name, aname)
        if !isempty(groups) && groups[end][1] == ftype && groups[end][2] == attr
            push!(groups[end][3], aname)
        else
            push!(groups, (ftype, attr, String[aname]))
        end
    end

    twidth = isempty(groups) ? 0 : maximum(length(ftype) + 1 for (ftype, _, _) in groups)
    awidth = isempty(groups) ? 0 : maximum(length(attr) for (_, attr, _) in groups)

    # `function foo(a, b) &` / `        bind(c, name='foo') result(ret)`
    for l in wrapped_call("$(indent)$kind $name", [a for (a, _) in args], " &")
        println(io, l)
    end
    suffix = retinfo === nothing ? "" : " result($(retinfo[2]))"
    println(io, "$(indent)    bind(c, name='$name')$suffix")

    # `use` first, then `import`, as the standard requires.
    isempty(args) || println(io, "$(body)use iso_c_binding")
    if retinfo !== nothing && startswith(retinfo[1], "type(Smc")
        println(io, "$(body)import :: SmcColoringOptions")
    end
    for (ftype, attr, names) in groups
        println(
            io,
            "$(body)$(rpad(ftype * ",", twidth)) $(rpad(attr, awidth)) :: $(join(names, ", "))",
        )
    end
    if retinfo !== nothing
        println(io, "$(body)$(rpad(retinfo[1], twidth + 1 + awidth)) :: $(retinfo[2])")
    end
    return println(io, "$(indent)end $kind $name")
end

open(OUT_PATH_F90, "w") do io
    println(
        io,
        """
! smc.f90 - Fortran interface to SparseMatrixColorings.jl
!
! Generated by interfaces/scripts/generate_header.jl from the same table as
! smc.h -- do not edit by hand, edit the generator instead.
!
! An include file rather than a module, so that no .mod file has to be shipped
! and any Fortran compiler can consume it:
!
!   program my_prog
!     use iso_c_binding
!     implicit none
!     include 'smc.f90'    ! <- here, after implicit none
!     ...
!   end program
!
! Every C pointer is a  type(c_ptr), value  dummy argument.  Pass c_loc(x)
! for an array or a struct you own -- it must carry the target attribute --
! or c_null_ptr where the C interface accepts NULL.  The single exception is
! the void** out-parameter of smc_coloring, declared type(c_ptr),
! intent(out), which receives the opaque result handle.
!
! Dense matrices are column-major, which is already Fortran's own layout, so
! a 2D array can be handed over directly with c_loc.  Buffer lengths are
! element counts, never byte counts.""",
    )

    println(io)
    println(io, "  ! Version")
    for (suffix, value) in
        [("MAJOR", _SV.major), ("MINOR", _SV.minor), ("PATCH", _SV.patch)]
        println(io, "  integer(c_int), parameter :: SMC_VERSION_$suffix = $value")
    end
    println(io)

    emit_f_banner(io, "  ", ["Enumerators  (must match smc.h)"])
    for (type_name, doc, entries) in ENUM_TABLES
        println(io)
        println(io, "  ! $type_name")
        emit_f_doc(io, "  ", doc)
        width = maximum(length(name) for (name, _) in entries)
        for (name, value) in entries
            println(io, "  integer(c_int), parameter :: $(rpad(name, width)) = $value")
        end
    end
    println(io)

    combo_lines = ["  $(rpad(s, 16)) $(rpad(p, 17)) $d" for (s, p, d) in SUPPORTED_COMBOS]
    emit_f_banner(
        io,
        "  ",
        [
            "Coloring options  (must match the struct in smc.h)",
            "",
            "Passed to smc_coloring and smc_fast_coloring as c_loc(opts), and",
            "remembered by the result handle.  Initialise with smc_default_options()",
            "before overriding individual fields; c_null_ptr means the defaults.",
            "",
            "Supported (structure, partition, decompression) combinations; anything",
            "else is rejected with -2:",
            combo_lines...,
        ],
    )
    println(io)

    # The 8 fields in the exact order of the C struct; continuation lines of the
    # per-field comments have an empty name in OPTION_FIELDS.
    nfields = count(!isempty(name) for (_, name, _) in OPTION_FIELDS)
    nfields == 8 || error("expected 8 SmcColoringOptions fields, found $nfields")
    println(io, "  type, bind(c) :: SmcColoringOptions")
    fdecls = [
        isempty(name) ? "" : "integer(c_int) :: $name" for (_, name, _) in OPTION_FIELDS
    ]
    fwidth = maximum(length, fdecls)
    for ((_, _, comment), decl) in zip(OPTION_FIELDS, fdecls)
        println(io, rstrip("    $(rpad(decl, fwidth))  ! $comment"))
    end
    println(io, "  end type SmcColoringOptions")
    println(io)

    emit_f_banner(
        io,
        "  ",
        [
            "Return codes",
            "",
            "Every function returning integer(c_int) returns one of:",
            "",
            "   0  success",
            "  -1  internal error (a Julia exception was caught and logged)",
            "  -2  unsupported combination of (structure, partition, decompression,",
            "      dtype)",
            "  -3  invalid argument (NULL pointer, bad dimension, buffer too small,",
            "      bad enum value, bad index_base)",
            "  -4  invalid or already-freed handle",
        ],
    )
    println(io)

    emit_f_banner(io, "  ", ["C function interfaces"])
    println(io)
    println(io, "  interface")

    sections = Dict(SECTIONS)
    for (i, (name, ret, args)) in enumerate(SIGS)
        println(io)
        if haskey(sections, name)
            emit_f_banner(io, "    ", sections[name])
            println(io)
        end
        println(io, f_comment("    ", F_RULE))
        println(io, f_comment("    ", name))
        println(io, f_comment("    ", ""))
        if haskey(FORTRAN_DOCS, name)
            emit_f_doc(io, "    ", FORTRAN_DOCS[name])
        else
            @warn "no documentation for $name in FORTRAN_DOCS; emitting a bare interface"
        end
        println(io, f_comment("    ", F_RULE))
        emit_f_interface(io, name, ret, args)
    end

    println(io)
    return println(io, "  end interface")
end

println("Generated $OUT_PATH_F90")
