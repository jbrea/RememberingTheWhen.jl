module NeuralProcessingOfTime
using Random


###
### Stimuli
###

@enum Color black red green blue orange pink yellow
@enum Shape shapeless triangle square pentagon circle halfcircle quartercircle

struct Token{C,S}
    c::C
    s::S
end
similarity(t1::Token, t2::Token) = ((t1.c == t2.c) + (t1.s == t2.s))/2
csimilarity(t1::Token, t2::Token) = (t1.c == t2.c) - (t1.s == 0) # ignore tokens with s == 0
function age_at_similarity(history, query; threshold = 1, similarity = similarity)
    age = 1
    for token in reverse(history)
        similarity(token, query) ≥ threshold && return age
        age += 1
    end
    return nothing
end

###
### Codes
###

Base.@kwdef mutable struct Rate
    r::Float64 = 0
    r₋::Float64 = r
end
function activity(r::Rate, i; previous = false)
    i != 1 && error("Rate neuron has only index 1; got $i.")
    previous ? r.r₋ : r.r
end
function setactivity!(r::Rate, i, v; previous = false)
    i != 1 && error("Rate neuron has only index 1; got $i.")
    if previous
        r.r₋ = v
    else
        r.r = v
    end
end
Base.length(::Rate) = 1
function sense!(r::Rate, v)
#     setactivity!(r, 1, v, previous = false)
    setactivity!(r, 1, v, previous = true)
end
@inline function update!(r::Rate, f)
    r.r₋ = f(r.r)
    r.r = 0.
    r
end
Base.@kwdef mutable struct OneHot{N}
    i::Int = 0
    i₋::Int = i
end
activity(o::OneHot, i; previous = false) = (previous ? o.i₋ : o.i) == i
function setactivity!(o::OneHot, i, v; previous = false)
    v != 1 && @warn "Cannot assign value $v to one-hot coded layer"
    if previous
        o.i₋ = i
    else
        o.i = i
    end
end
Base.length(::OneHot{N}) where N = N
function sense!(o::OneHot, i)
#     setactivity!(o, Int(i), 1, previous = false)
    setactivity!(o, Int(i), 1, previous = true)
end
@inline function update!(o::OneHot, ::Any)
    o.i₋ = o.i
    o.i = 0
    o
end
Base.@kwdef struct Distributed
    x::Vector{Float64}
    x₋::Vector{Float64} = copy(x)
end
Distributed(n::Int) = Distributed(x = zeros(n))
activity(d::Distributed, i; previous = false) = previous ? d.x₋[i] : d.x[i]
function setactivity!(d::Distributed, i, v; previous = false)
    if previous
        d.x₋[i] = v
    else
        d.x[i] = v
    end
end
Base.length(d::Distributed) = length(d.x)
@inline function update!(d::Distributed, f)
    x₋, x = d.x₋, d.x
    for i in eachindex(x)
        x₋[i] = f(x[i])
        x[i] = 0
    end
    d
end
# struct OrderedPopRate
#     x::Distributed
# end
# activity(d::OrderedPopRate, i) = activity(d.x, i)
# update!(o::OrderedPopRate) = update!(o.x)
# Base.length(o::OrderedPopRate) = length(o.x)
heaviside(x) = ifelse(x > 0, 1., 0.)
struct Neurons{C, A}
    id::String
    code::C
    activation::A
end
function Base.show(io::IO, n::Neurons{C, A}) where {C, A}
    print(io, "Neurons ($(n.id), $C, $A)")
end
@inline update!(n::Neurons) = update!(n.code, n.activation)
activity(n::Neurons; previous = false) = [activity(n, i; previous) for i in 1:length(n)]
function activity(n::Neurons, i; previous = false)
    # for previous activity activation is already applied in update!
    # TODO: This looks like a horrible hack and a source for bugs (e.g. in RewardModulatedHebbian using post instead of post.code can lead to wrong results)
    if previous
        activity(n.code, i, previous = true)
    else
        n.activation(activity(n.code, i, previous = false))
    end
end
function setactivity!(n::Neurons, i, v; previous = false)
    setactivity!(n.code, i, v; previous)
end
Base.length(n::Neurons) = length(n.code)
Neurons(id, code) = Neurons(id, code, identity)
is_silent(::Any) = false
function mul(w, i, pre::OneHot, offset = 0)
    pre.i₋ == 0 && return 0.
    w[i, pre.i₋ + offset]
end
mul(w, i, pre::Rate, offset = 0) = w[i, 1 + offset] * pre.r₋
mul(w, i, pre::Distributed, offset = 0) = sum(w[i, j + offset] * pre.x₋[j]
                                              for j in eachindex(pre.x₋))
function propagate!(post::Neurons{<:Distributed}, pre, w, with_bias)
    for i in eachindex(post.code.x)
        post.code.x[i] += mul(w, i, pre.code)
        if with_bias
            post.code.x[i] += w[i, end]
        end
    end
end
function propagate!(post, pre, w, with_bias, modulator)
    is_silent(modulator) && return
    propagate!(post, pre, w, with_bias)
end

###
### Associations
###

struct Concat{T1, T2}
    x::T1
    y::T2
end
function activity(c::Concat, i; previous = false)
    Nx = length(c.x)
    if i ≤ Nx
        activity(c.x, i; previous)
    else
        activity(c.y, i - Nx; previous)
    end
end
Base.length(c::Concat) = length(c.x) + length(c.y)
struct Product{T1, T2}
    x::T1
    y::T2
end
function update!(c::Union{Concat, Product}, f)
    update!(c.x, f)
    update!(c.y, f)
end
function activity(p::Product, i; previous = false)
    Nx = length(p.x)
    ix, iy = divrem(i, Nx)
    activity(p.x, ix + 1; previous) * activity(p.y, iy + 1; previous)
end
Base.length(c::Product) = length(c.x) * length(c.y)
mul(w, i, pre::Concat) = mul(w, i, pre.x) + mul(w, i, pre.y, length(pre.x))
mul(w, i, pre::Product{<:OneHot, <:OneHot}) = w[i, pre.x.i₋ * length(pre.x) + pre.y.i₋]
function mul(w, i, pre::Product) # fallback
    sum(w[i, j] * activity(pre, j; previous = true) for j in 1:length(pre))
end

###
### Plasticity
###

Base.@kwdef struct Hebbian
    η::Float64 = 1
end
_getindex(x::Number, ::Any, Any) = x
_getindex(x::AbstractMatrix, i, j) = x[i, j]
_clamp!(::Any, ::Nothing, ::Nothing, ::Any, ::Any) = nothing
@inline function _clamp!(w, ::Nothing, max, i, j)
    m = _getindex(max, i, j)
    if w[i, j] > m
        w[i, j] = m
    end
end
@inline function _clamp!(w, min, ::Nothing, i, j)
    m = _getindex(min, i, j)
    if w[i, j] < m
        w[i, j] = m
    end
end
@inline function _clamp!(w, min, max, i, j)
    _clamp!(w, nothing, max, i, j)
    _clamp!(w, min, nothing, i, j)
end
function hebbian_update!(w, η, post, pre, modulator = 1.;
                         previous_post = false, previous_pre = false,
                         min = nothing, max = nothing)
    N_pre = length(pre)
    for j in axes(w, 2), i in axes(w, 1)
        _post = activity(post, i, previous = previous_post)
        _pre = j > N_pre ? 1. : activity(pre, j, previous = previous_pre)
#         if _pre ≠ 0 && _post ≠ 0
#             @show (_pre, _post)
#         end
        w[i, j] += η * modulation(modulator, i, j) * _post * _pre
        _clamp!(w, min, max, i, j)
    end
end
update!(w, post, pre, h::Hebbian) = hebbian_update!(w, h.η, post, pre)
Base.@kwdef struct ModulatedHebbian{M}
    modulator::M
    η::Float64 = 1.
end
modulation(::Any, ::Any, ::Any) = 1.
modulation(m::AbstractMatrix, i, j) = m[i, j]
modulation(m::Number, ::Any, ::Any) = m
function update!(w, post, pre, m::ModulatedHebbian)
    modulator = modulation(m.modulator)
    modulator == 0 && return
    hebbian_update!(w, m.η, post, pre, modulator)
end
struct StrengthDecay{K}
    kind::K
end
update!(w, ::Any, ::Any, p::StrengthDecay) = update!(w, p.kind)
struct Additive
    Δ::Float64
end
struct Multiplicative
    factor::Float64
end
function update!(w::AbstractMatrix, kind::Additive)
    w .-= kind.Δ
    clamp!(w, 0, Inf)
end
function update!(w::AbstractMatrix, kind::Multiplicative)
    w .*= kind.factor
end
struct HebbianLatentStateDecay{K}
    latent_state::Matrix{Float64}
    latent_state_increment::Matrix{Float64}
    θ::Float64
    kind::K
    η::Float64
end
function HebbianLatentStateDecay(; post = nothing, pre = nothing,
                                   latent_state = zeros(length(post), length(pre)),
                                   latent_state_increment = Float64[i for i in 1:length(post), j in 1:length(pre)],
                                   θ = 1e-3, kind = Additive(1.), η = 1.)
    HebbianLatentStateDecay(latent_state,
                            latent_state_increment,
                            θ, kind, η)
end
function _set_weights!(w, latent_state, θ)
    @inbounds for i in eachindex(w)
        if latent_state[i] < θ
            w[i] = latent_state[i] = 0
        else
            w[i] = 1.
        end
    end
end
function update!(w, post, pre, p::HebbianLatentStateDecay)
    _set_weights!(w, p.latent_state, p.θ)
    hebbian_update!(p.latent_state, p.η, post, pre, p.latent_state_increment,
                    max = p.latent_state_increment) # replacing traces
#     @show p.latent_state post.code.x pre.code.x
    update!(p.latent_state, p.kind)
end
struct RewardModulatedHebbian{R,C,B}
    eligibility::Matrix{Float64}
    reward_sensor::R
    η::Float64
    η_bias::B
    clipper::C
end
Base.@kwdef struct Clamp
    lo::Float64 = 0
    hi::Float64 = Inf
end
(c::Clamp)(w) = clamp!(w, c.lo, c.hi)
struct Shifter end
function (s::Shifter)(w)
    miw = minimum(w)
    if miw < 0
        w .-= miw
    end
end
function RewardModulatedHebbian(; post = nothing, pre = nothing,
                                  with_bias = false,
                                  eligibility = zeros(length(post), length(pre) + with_bias),
                                  reward_sensor = RewardSensor(), η = 1., η_bias = with_bias ? η : nothing,
                                  clipper = Clamp())
    RewardModulatedHebbian(eligibility, reward_sensor, η, η_bias, clipper)
end
_adjust_bias!(::Nothing, ::Any, ::Any) = nothing
_adjust_bias!(η_bias, η, eligibility) = eligibility[:, end] .*= η_bias / η
function update!(w, post, pre, r::RewardModulatedHebbian)
#     println("eligibility")
#     display(r.eligibility)
    R = reward(r.reward_sensor)
#     if R ≠ 0 || sum(r.eligibility) > 0
#         @show R r.eligibility
#     end
#     if post.code.x[1] > 0
#         hebbian_update!(r.eligibility, 1., post, pre, previous_pre = true)
#         @show post.code pre.code r.eligibility
#     end
    _adjust_bias!(r.η_bias, r.η, r.eligibility)
#     if R ≠ 0
#         @show R r.eligibility
#     end
    w .+= r.η * R * r.eligibility
    r.clipper(w)
    r.eligibility .= 0
    hebbian_update!(r.eligibility, 1., post.code, pre, previous_pre = true)
end
struct DelayedHebbian
    eligibility::Matrix{Float64}
    η::Float64
end
function DelayedHebbian(; post = nothing, pre = nothing,
                          eligibility = zeros(length(post), length(pre)), η = 1.)
    DelayedHebbian(eligibility, η)
end
function update!(w, post, pre, r::DelayedHebbian)
    w .+= r.η * r.eligibility
    r.eligibility .= 0
    hebbian_update!(r.eligibility, 1., post, pre)
end
# TODO: more bio-plausible
struct IncrementPost{M}
    modulator::M
    steps::Int
end
function update!(w, ::Any, ::Any, i::IncrementPost)
    modulation(i.modulator) == 0 && return
    for k in reverse(axes(w, 1)), j in axes(w, 2)
        if k%i.steps == 1
            w[k, j] = 0
        else
            w[k, j] = w[k-1, j]
        end
    end
end
# struct CopyFrom{C}
#     from::C
# end
# function update!(w, ::Any, ::Any, CopyFrom)
#     for i in axes(w, 1), j in axes(w, 2)
#         w[i, j] = from.w[i, j]
#         from.w[i, j]
#     end
# end

###
### Brains
###
abstract type AbstractConnection end
abstract type AbstractStaticConnection{M} <: AbstractConnection end
Base.@kwdef struct Connection{P, M, I, O} <: AbstractConnection
    plasticity::P = tuple()
    modulator::M = nothing
    with_bias::Bool = false
    pre::I
    post::O
    w::Matrix{Float64} = zeros(length(post), length(pre) + with_bias)
end
_showprepost(io, c) = print(io, "$(c.pre.id) -> $(c.post.id) : $(typeof(c).name.name)")
function Base.show(io::IO, c::AbstractStaticConnection{M}) where M
    _showprepost(io, c)
    if M != Nothing
        print(io, " (")
        show(io, M)
        print(io, ")")
    end
end
function Base.show(io::IO, c::Connection{P,M}) where {P,M}
    _showprepost(io, c)
    print(io,  " w/$(c.with_bias ? "" : "o") bias")
    if M != Nothing || P != Nothing
        print(io, " (")
        if P != Nothing
            show(io, P)
            if M != Nothing
                print(io, ", ")
            end
        end
        if M != Nothing
            show(io, M)
        end
        print(io, ")")
    end
end
propagate!(c::Connection) = propagate!(c.post, c.pre, c.w, c.with_bias, c.modulator)
function update!(c::Connection)
    for p in c.plasticity
        update!(c.w, c.post, c.pre, p)
    end
end
Base.@kwdef struct One2OneConnection{M, I, O} <: AbstractStaticConnection{M}
    modulator::M = nothing
    pre::I
    post::O
end
indices(c::One2OneConnection) = zip(1:length(c.pre), 1:length(c.post))
update!(::One2OneConnection) = nothing
function propagate!(o::One2OneConnection)
    post = o.post
    for i in eachindex(post.code.x)
        setactivity!(post.code, i, activity(o.pre.code, i, previous = true))
    end
end
Base.@kwdef struct All2AllConnection{M, I, O} <: AbstractStaticConnection{M}
    modulator::M = nothing
    pre::I
    post::O
end
indices(c::Union{Connection, All2AllConnection}) = Iterators.product(1:length(c.pre), 1:length(c.post))
function propagate!(o::All2AllConnection)
    post = o.post
    for i in eachindex(post.code.x)
        v = 0.
        for j in 1:length(o.pre.code)
            v += activity(o.pre.code, j, previous = true)
        end
        setactivity!(post.code, i, v)
    end
end
update!(::All2AllConnection) = nothing
Base.@kwdef struct All2FirstConnection{M, I, O} <: AbstractStaticConnection{M}
    modulator::M = nothing
    pre::I
    post::O
end
indices(c::All2FirstConnection) = Iterators.product(1:length(c.pre), 1:1)
# Base.show(io::IO, ::All2FirstConnection{M}) where M = print(io, "All2FirstConnection ($M)")
function propagate!(o::All2FirstConnection)
    post = o.post
    v = 0.
    for j in 1:length(o.pre.code)
        v += activity(o.pre.code, j, previous = true)
    end
    setactivity!(post.code, 1, v)
end
update!(::All2FirstConnection) = nothing
Base.@kwdef struct All2FirstOfKindConnection{M, I, O} <: AbstractStaticConnection{M}
    modulator::M = nothing
    pre::I
    post::O
    off::Int
end
indices(c::All2FirstOfKindConnection) = [(i, (i-1)*c.off+1) for i in 1:length(c.pre)]
# Base.show(io::IO, ::All2FirstOfKindConnection{M}) where M = print(io, "All2FirstOfKindConnection ($M)")
function propagate!(o::All2FirstOfKindConnection)
    post = o.post
    for j in 1:length(o.pre.code)
        setactivity!(post.code, (j-1)*o.off + 1, activity(o.pre.code, j, previous = true))
    end
end
update!(::All2FirstOfKindConnection) = nothing
struct SparseRandomConnection{M, I, O} <: AbstractStaticConnection{M}
    modulator::M
    idxs::Vector{Vector{Int}}
    pre::I
    post::O
end
indices(c::SparseRandomConnection) = vcat([[(i, j) for i in c.idxs[j]] for j in 1:length(c.post)]...)
function SparseRandomConnection(; modulator = nothing, pre, post, sparsity = 0.1, weights = nothing,
                                  idxs = sparse_random_connections(pre, post; sparsity, weights))
    SparseRandomConnection(modulator, idxs, pre, post)
end
Base.@kwdef struct RandomInFan
    min::Int = 1
    max::Int
end
function k_out_of_n(k::Int, n, weights)
    indices = collect(1:n)
    _weights = copy(weights)
    selected_indices = zeros(Int, k)
    for i in eachindex(selected_indices)
        chosen_index = wsample(_weights)
        selected_indices[i] = indices[chosen_index]
        deleteat!(indices, chosen_index)
        deleteat!(_weights, chosen_index)
    end
    selected_indices
end
k_out_of_n(k::Int, n, ::Nothing) = randperm(n)[1:k]
k_out_of_n(::Val{:uniform}, n, weights) = k_out_of_n(rand(1:n), n, weights)
k_out_of_n(r::RandomInFan, n, weights) = k_out_of_n(rand(r.min:r.max), n, weights)
function sparse_random_connections(pre, post;
                                   sparsity = 0.1,
                                   weights = nothing,
                                   nin = isa(sparsity, Number) ? floor(Int, length(pre)*sparsity) : sparsity)
    [k_out_of_n(nin, length(pre), weights) for _ in 1:length(post)]
end
# Base.show(io::IO, ::SparseRandomConnection{M}) where M = print(io, "SparseRandomConnection ($M)")
function propagate!(o::SparseRandomConnection)
    post = o.post
    for i in eachindex(post.code.x)
        v = 0.
        for j in o.idxs[i]
            v += activity(o.pre.code, j, previous = true)
        end
        setactivity!(post.code, i, v)
    end
end
update!(::SparseRandomConnection) = nothing
struct TokenSensor{C}
    neurons::C
end
function TokenSensor(; n_color = length(instances(Color)),
                       n_shape = length(instances(Shape)),
                       color_code = OneHot{n_color-1}(),
                       shape_code = OneHot{n_shape-1}(),
                       association = Concat)
    TokenSensor(Neurons("token sensor", association(color_code, shape_code)))
end
function sense!(s::TokenSensor, token)
    sense!(s.neurons.code.x, token.c)
    sense!(s.neurons.code.y, token.s)
    s
end
struct RewardSensor{C}
    neurons::C
end
reward(r::RewardSensor) = activity(r.neurons, 1, previous = true)
RewardSensor(; code = Rate()) = RewardSensor(Neurons("reward sensor", code))
sense!(r::RewardSensor, reward) = sense!(r.neurons.code, reward)
Base.@kwdef mutable struct FixedMicroSteps
    N::Int = 2
    i::Int = 0
end
function (f::FixedMicroSteps)()
    f.i += 1
    if f.i > f.N
        false
    else
        true
    end
end
reset!(f::FixedMicroSteps) = f.i = 0
modulation(f::FixedMicroSteps) = f.i == f.N
Base.@kwdef struct Brain{C, N, A, S}
    token_sensor::TokenSensor = TokenSensor()
    reward_sensor::RewardSensor = RewardSensor()
    connections::C
    actuators::A
    neurons::N = collect_neurons(token_sensor, reward_sensor, connections..., actuators)
    is_micro_step!::S = FixedMicroSteps()
end
struct OnlyActiveAtStep
    i::Int
    stepper::FixedMicroSteps
end
is_silent(o::OnlyActiveAtStep) = o.i != o.stepper.i
function Base.show(io::IO, brain::Brain)
    print(io, "Brain")
    for connection in brain.connections
        print(io, "\n  $(connection)")
    end
end
function collect_neurons(args...)
    neurons = Neurons[]
    for arg in args
        fns = fieldnames(typeof(arg))
        for fn in fns
            f = getfield(arg, fn)
            typeof(f) <: Neurons && push!(neurons, f)
        end
    end
    tuple(unique(neurons)...)
end
@Base.kwdef struct IdPlus
    b::Float64 = 1e-5
end
(f::IdPlus)(x) = x + f.b
postaction_activity_winner(::typeof(exp), sa, S) = 1 - sa/S
postaction_activity_looser(::typeof(exp), sa, S) = - sa/S
postaction_activity_winner(::IdPlus, sa, S) = 1/sa - 1/S
postaction_activity_looser(::IdPlus, sa, S) = -1/S
postaction_activity_winner(::typeof(identity), sa, S) = 1.
postaction_activity_looser(::typeof(identity), sa, S) = 0.
_numerically_stabilize!(::Any, ::Any) = nothing
_numerically_stabilize!(code, ::typeof(exp)) = code .-= maximum(code)
function action!(actuator; rng = Random.GLOBAL_RNG, greedy = false)
    Na = length(actuator)
    S0 = sum(actuator.code.x[i] for i in 1:Na)
    S0 == 0 && return 0 # Is this needed anywhere?
    _numerically_stabilize!(actuator.code.x, actuator.activation)
    S = sum(activity(actuator, i) for i in 1:Na)
    awinner = 0
    if greedy
        awinner = argmax(activity(actuator, i) for i in 1:Na)
    else
        θ = rand(rng) * S
    #     @show [activity(actuator, i) for i in 1:Na] θ
        s = 0.
        for a in 1:Na
            sa = activity(actuator, a)
            s += sa
            if s > θ
                awinner = a
                break
            end
        end
    end
    for a in 1:Na
        sa = activity(actuator, a)
        if a == awinner
#             @show (awinner, sa, S, postaction_activity_winner(actuator.activation, sa, S))
            setactivity!(actuator, a, postaction_activity_winner(actuator.activation, sa, S))
        else
#             @show (a, sa, S, postaction_activity_looser(actuator.activation, sa, S))
            setactivity!(actuator, a, postaction_activity_looser(actuator.activation, sa, S))
        end
    end
#     @show actuator.code.x
    return awinner
end

function wsample(weights; rng = Random.GLOBAL_RNG)
    s = sum(weights)
    θ = rand(rng) * s
    s = 0.
    for i in eachindex(weights)
        s += weights[i]
        s > θ && return i
    end
end

update!(::Tuple{}) = nothing
function update!(x::Tuple)
    update!(first(x))
    update!(Base.tail(x))
end
propagate!(::Tuple{}) = nothing
function propagate!(x::Tuple)
    propagate!(first(x))
    propagate!(Base.tail(x))
end


function micro_step!(brain::Brain; callback = () -> nothing, greedy = false)
    propagate!(brain.connections)
#     @show brain.neurons[end-1].code
    a = action!(brain.actuators; greedy)
    update!(brain.connections)
    update!(brain.neurons)
#     @show brain.neurons[end-1].code
#     @show activity.(Ref(brain.neurons[end-1]), 1:length(brain.neurons[end-1]),
#                     previous = true)
    a
end

function step!(brain, token, reward; callback = () -> nothing, greedy = false)
    sense!(brain.token_sensor, token)
    sense!(brain.reward_sensor, reward)
    callback()
    while brain.is_micro_step!()
        a = micro_step!(brain; callback, greedy)
#         @show brain.actuators.code
        callback()
        if a > 0
            reset!(brain.is_micro_step!)
            return a
        end
    end
    reset!(brain.is_micro_step!)
    return 0
end


end # module
