mutable struct Forest{T<:Integer}
    counter::T
    intmap::Dict{Tuple{T,T},T}
    parents::Vector{T}
    ranks::Vector{T}
    ntrees::T
end

function Forest{T}(n::Integer) where {T<:Integer}
    counter = zero(T)
    intmap = Dict{Tuple{T,T},T}()
    sizehint!(intmap, n)
    parents = collect(Base.OneTo(T(n)))
    ranks = zeros(T, T(n))
    ntrees = T(n)
    return Forest{T}(counter, intmap, parents, ranks, ntrees)
end

function Base.push!(forest::Forest{T}, edge::Tuple{T,T}) where {T<:Integer}
    forest.counter += 1
    forest.intmap[edge] = forest.counter
    forest.ntrees += one(T)
    return edge
end

function _find_root!(parents::Vector{T}, index_edge::T) where {T<:Integer}
    @inbounds p = parents[index_edge]
    @inbounds if parents[p] != p
        parents[index_edge] = p = _find_root!(parents, p)
    end
    return p
end

function find_root!(forest::Forest{T}, edge::Tuple{T,T}) where {T<:Integer}
    return _find_root!(forest.parents, forest.intmap[edge])
end

function root_union!(forest::Forest{T}, index_edge1::T, index_edge2::T) where {T<:Integer}
    parents = forest.parents
    rks = forest.ranks
    @inbounds rank1 = rks[index_edge1]
    @inbounds rank2 = rks[index_edge2]

    if rank1 < rank2
        index_edge1, index_edge2 = index_edge2, index_edge1
    elseif rank1 == rank2
        rks[index_edge1] += one(T)
    end
    @inbounds parents[index_edge2] = index_edge1
    forest.ntrees -= one(T)
    return nothing
end
