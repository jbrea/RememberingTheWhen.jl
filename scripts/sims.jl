using NeuralProcessingOfTime: similarity
using NeuralProcessingOfTime
using DataFrames, PGFPlotsX, Statistics, ColorSchemes, Distributions
import NeuralProcessingOfTime: age_at_similarity
import NeuralProcessingOfTime: Token, Color, Shape, age_at_similarity, IdPlus, Clamp, Shifter
import NeuralProcessingOfTime: Neurons, Distributed, Connection, TokenSensor, Hebbian,
       One2OneConnection, All2AllConnection, HebbianLatentStateDecay, Brain, RewardSensor,
       RewardModulatedHebbian, FixedMicroSteps, OnlyActiveAtStep, DelayedHebbian,
       Additive, IncrementPost, All2FirstConnection, SparseRandomConnection,
       All2FirstOfKindConnection
import NeuralProcessingOfTime: sense!, micro_step!, update!, propagate!, activity,
       setactivity!, heaviside, step!, action!, RandomInFan
import NeuralProcessingOfTime: black, red, green, blue, orange, pink, yellow,
       shapeless, triangle, square, pentagon, circle, halfcircle, quartercircle
const colors = ColorSchemes.Johnson
using Random

const BASEPATH = joinpath(@__DIR__, "..")
const DOCPATH = joinpath(BASEPATH, "doc")
const DATAPATH = joinpath(BASEPATH, "data")

using Serialization, CodecZstd
function csave(f, d)
    open(f, "w") do fd
        stream = ZstdCompressorStream(fd)
        serialize(stream, d)
        close(stream)
    end
end
function cload(f)
    open(f, "r") do fd
        stream = ZstdDecompressorStream(fd)
        res = deserialize(stream)
        close(stream)
        res
    end
end

function reward_task1(history, query, action)
    age = age_at_similarity(history, query, threshold = 1)
    age != 3 && return 0.
    action == 1 && return -1.
    return 1.
end
function reward_task2(history, query, action)
    age = age_at_similarity(history, query, threshold = 1)
    (age ∉ (2, 3) || query.c ∉ (blue, red)) && return 0.
    if age == 2
        query.c == blue && action == 2 && return 1.
        query.c == red && action == 1 && return 1.
    elseif age == 3
        query.c == blue && action == 1 && return 1.
        query.c == red && action == 2 && return 1.
    end
    return -1.
end
function reward_task3(history, query, action)
    age = age_at_similarity(history, query, threshold = 1)
    (age === nothing || age > 5 || query.s === circle) && return 0.
    ((age ≤ 2 && action == 1) ||
     (age > 2 && action == 2)) && return 1.
     return -1.
end

function stimuli_task1(; test_interval = rand(1:5), version = 1)
    tokens = [Token(red, triangle),
              Token(blue, square),
              Token(green, circle),
              Token(yellow, pentagon),
              Token(orange, halfcircle)]
    sequence = [tokens[1:3]; tokens[1]]
    if test_interval == 1
        append!(sequence, [tokens[4], tokens[4]])
    elseif test_interval == 2
        push!(sequence, tokens[3])
    elseif test_interval == 3
        if version == 1
            push!(sequence, tokens[2])
        else
            append!(sequence, [tokens[4], tokens[3]])
        end
    elseif test_interval == 4
        append!(sequence, [tokens[4], tokens[2]])
    elseif test_interval == 5
        append!(sequence, [tokens[4:5]; tokens[2]])
    end
    sequence
end
function stimuli_task2(; trials = 50, inter_trial = 5)
    tokens = [Token(red, triangle),
              Token(black, circle),
              Token(blue, square)]
    sequence = Token{Color, Shape}[]
    for t in 1:trials
        tok = iseven(t) ? tokens[1] : tokens[3]
        for filler in (1, 2)
            push!(sequence, tok)
            for _ in 1:filler
                push!(sequence, tokens[2])
            end
            push!(sequence, tok)
            for _ in 1:inter_trial # inter trial interval
                push!(sequence, tokens[2])
            end
        end
    end
    sequence
end
function stimuli_task3(; trials = 50,
                         max_age = 5,
                         inter_trial = max_age,
                         max_length = 10*trials,
                         rng = Random.GLOBAL_RNG)
    tokens = [Token(red, triangle),
              Token(black, circle)
             ]
    sequence = Token{Color, Shape}[]
    for _ in 1:trials
        length(sequence) == max_length && return sequence
        push!(sequence, tokens[1])
        RI = rand(rng, 0:max_age-1)
        for _ in 1:RI
            length(sequence) == max_length && return sequence
            push!(sequence, tokens[2])
        end
        push!(sequence, tokens[1])
        for _ in 1:inter_trial
            push!(sequence, tokens[2])
        end
    end
    for _ in 1:max_length - length(sequence)
        push!(sequence, tokens[2])
    end
    sequence
end

struct RandomBrain end
step!(::RandomBrain, ::Any, ::Any; kwargs...) = rand((1, 2))
mutable struct OptimalLearner{S}
    history::Vector{Token}
    STM::S
    Q::Dict{S, Float64}
end
function OptimalLearner(; only_age = false,
                          sinit = only_age ? (1, 1) : (Token(black, square), 1, 1))
    OptimalLearner(Token[],
                   sinit,
                   Dict{typeof(sinit), Float64}())
end
query(::OptimalLearner{<:Tuple{<:Token,Int,Int}}, token, age, a) = (token, age, a)
query(::OptimalLearner{Tuple{Int,Int}}, token, age, a) = (age, a)
function step!(o::OptimalLearner, token, reward; kwargs...)
    age = age_at_similarity(o.history, token, threshold = 1)
    if age === nothing
        age = -1
    end
    push!(o.history, token)
    o.Q[o.STM] = reward
    qs = [get(o.Q, query(o, token, age, a), 0.) for a in 1:2]
    a = rand(findall(==(maximum(qs)), qs))
    o.STM = query(o, token, age, a)
    return a
end

mutable struct Tracker1
    a::Int
    r::Vector{Float64}
end
Tracker1() = Tracker1(0, [])
function (t::Tracker1)(a, r)
    t.a = a
    r != 0 && push!(t.r, r)
end
struct TrackAll
    a::Vector{Int}
    r::Vector{Float64}
end
TrackAll() = TrackAll([], [])
(t::TrackAll)(a, r) = (push!(t.a, a); push!(t.r, r))
function run_task1!(brain, history, tracker = Tracker1(); callback = () -> nothing)
    t = run_task!(brain, history, reward_task1, tracker; callback)
    t.a, t.r
end
function run_task2!(brain, history, tracker = TrackAll(); callback = () -> nothing)
    run_task!(brain, history, reward_task2, tracker; callback)
end
function run_task3!(brain, history, tracker = TrackAll(); callback = () -> nothing)
    run_task!(brain, history, reward_task3, tracker; callback)
end
function run_task!(brain, history, reward, tracker; callback = () -> nothing, greedy = false)
    r = 0.
    for i in eachindex(history)
        a = step!(brain, history[i], r; callback, greedy)
        r = reward(history[1:i-1], history[i], a)
        callback() == "x" && break
        tracker(a, r)
    end
    tracker
end

# HebbianLatentStateDecay
function initial_weights(n1, n2, w1, w2; only_first = false, with_bias = false)
    [only_first && j < length(n2) ? 0. : a == 1 ? w1 : w2 for a in 1:length(n1), j in 1:length(n2) + with_bias]
end
function hebbian_latent_state_decay_brain(; winit = [.9, .2],
                                            only_first = false,
                                            η = 1.,
                                            act = IdPlus(),
                                            clipper = Clamp())
    actuators = Neurons("actuators", Distributed(x = zeros(2)), act)
    token_sensor = TokenSensor()
    reward_sensor = RewardSensor()
    intermediate = Neurons("intermediate", Distributed(length(token_sensor.neurons)),
                           heaviside)
    content = Neurons("content", Distributed(length(token_sensor.neurons)), heaviside)
    tag = Neurons("tag", Distributed(6), heaviside)
    stepper = FixedMicroSteps(N = 3)
    connections = (
                   One2OneConnection(pre = token_sensor.neurons,
                                     post = intermediate),
                   One2OneConnection(pre = token_sensor.neurons,
                                     post = content),
                   All2AllConnection(pre = token_sensor.neurons,
                                     post = tag),
                   Connection(plasticity = (DelayedHebbian(pre = intermediate,
                                                           post = content), ),
                              pre = intermediate,
                              post = content),
                   Connection(plasticity = (HebbianLatentStateDecay(pre = intermediate,
                                                                    post = tag,
                                                                    kind = Additive(1/3)),
                                           ),
                              pre = intermediate,
                              post = tag),
                   Connection(plasticity = (RewardModulatedHebbian(; pre = tag,
                                                                     post = actuators,
                                                                     η, clipper,
                                                                     reward_sensor),),
                              pre = tag, post = actuators,
                              modulator = OnlyActiveAtStep(3, stepper),
                              w = initial_weights(actuators, tag, winit[1], winit[2]; only_first)
                             ),
                   Connection(plasticity = (RewardModulatedHebbian(; pre = content,
                                                                     post = actuators,
                                                                     η, clipper,
                                                                     reward_sensor),),
                              pre = content, post = actuators,
                              modulator = OnlyActiveAtStep(3, stepper),
                              w = initial_weights(actuators, content, .5, .5; only_first)
                             )
                  )
    Brain(; reward_sensor, token_sensor, connections, actuators,
            is_micro_step! = stepper)
end

function increment_post_brain(; winit = [.9, .2], only_first = false, η = 1., act = IdPlus(), clipper = Clamp())
    actuators = Neurons("actuators", Distributed(x = zeros(2)), act)
    token_sensor = TokenSensor()
    reward_sensor = RewardSensor()
    intermediate = Neurons("intermediate", Distributed(length(token_sensor.neurons)),
                           heaviside)
    content = Neurons("content", Distributed(length(token_sensor.neurons)), heaviside)
    tag = Neurons("tag", Distributed(6), heaviside)
    stepper = FixedMicroSteps(N = 3)
    connections = (
                   One2OneConnection(pre = token_sensor.neurons,
                                     post = intermediate),
                   One2OneConnection(pre = token_sensor.neurons,
                                     post = content),
                   All2FirstConnection(pre = token_sensor.neurons,
                                     post = tag),
                   Connection(plasticity = (DelayedHebbian(pre = intermediate,
                                                           post = content), ),
                              pre = intermediate,
                              post = content),
                   Connection(plasticity = (IncrementPost(stepper, 6),
                                            DelayedHebbian(pre = intermediate, post = tag)
                                           ),
                              pre = intermediate,
                              post = tag),
                   Connection(plasticity = (RewardModulatedHebbian(; pre = tag,
                                                                     post = actuators,
                                                                     η, clipper,
                                                                     reward_sensor),),
                              pre = tag, post = actuators,
                              modulator = OnlyActiveAtStep(3, stepper),
                              w = initial_weights(actuators, tag, winit[1], winit[2]; only_first)
                             ),
                   Connection(plasticity = (RewardModulatedHebbian(; pre = content,
                                                                     post = actuators,
                                                                     η, clipper,
                                                                     reward_sensor),),
                              pre = content, post = actuators,
                              modulator = OnlyActiveAtStep(3, stepper),
                              w = initial_weights(actuators, content, winit[1], winit[2]; only_first)
                             ),
                  )
    Brain(; reward_sensor, token_sensor, connections, actuators, is_micro_step! = stepper)
end

function chrono_brain(; winit = [.9, .2], only_first = false, η = 1., act = IdPlus(), clipper = Clamp())
    actuators = Neurons("actuators", Distributed(x = zeros(2)), act)
    token_sensor = TokenSensor()
    reward_sensor = RewardSensor()
    intermediate = Neurons("intermediate", Distributed(length(token_sensor.neurons)),
                           heaviside)
    chrono = Neurons("chrono", Distributed(6*length(token_sensor.neurons)), heaviside)
    stepper = FixedMicroSteps(N = 3)
    connections = (
                   One2OneConnection(pre = token_sensor.neurons,
                                     post = intermediate),
                   All2FirstOfKindConnection(pre = token_sensor.neurons,
                                             post = chrono,
                                             off = 6),
                   Connection(plasticity = (IncrementPost(stepper, 6),
                                            DelayedHebbian(pre = intermediate, post = chrono)
                                           ),
                              pre = intermediate,
                              post = chrono),
                   Connection(plasticity = (RewardModulatedHebbian(; pre = chrono,
                                                                     post = actuators,
                                                                     η, clipper,
                                                                     reward_sensor),),
                              pre = chrono, post = actuators,
                              modulator = OnlyActiveAtStep(3, stepper),
                              w = initial_weights(actuators, chrono, winit[1], winit[2]; only_first)
                             )
                  )
    Brain(; reward_sensor, token_sensor, connections, actuators, is_micro_step! = stepper)
end

function sparse_random_brain(; winit = [.9, .2], only_first = false, η = 1., η_bias = nothing,
                               act = IdPlus(), clipper = Clamp(),
                               n_intermediate = 50, n_content = 200, n_action = 2,
                               latent_state_increment_factor = 6,
                               n_color = length(instances(NeuralProcessingOfTime.Color)),
                               n_shape = length(instances(NeuralProcessingOfTime.Shape)),
                               sparsity = 1/(n_color+n_shape-2),
                               max_fanin = n_color+n_shape-2,
                               min_fanin = 1,
                               with_readout_bias = true,
                               latent_state_distribution = Uniform(0, latent_state_increment_factor),
                               in_weights = nothing,
                               content_activation = x -> heaviside(x - 1.5))
    actuators = Neurons("actuators", Distributed(x = zeros(n_action)), act)
    token_sensor = TokenSensor(; n_color, n_shape)
    reward_sensor = RewardSensor()
    intermediate = Neurons("intermediate", Distributed(n_intermediate), heaviside)
    content = Neurons("content", Distributed(n_content), content_activation)
    stepper = FixedMicroSteps(N = 3)
    connections = (
                   SparseRandomConnection(; pre = token_sensor.neurons,
                                            post = intermediate,
                                            sparsity),
                   SparseRandomConnection(pre = token_sensor.neurons,
                                          post = content,
                                          weights = in_weights,
                                          sparsity = RandomInFan(min = min_fanin, max = max_fanin)),
                   Connection(plasticity = (HebbianLatentStateDecay(pre = intermediate,
                                                                    post = content,
                                                                    kind = Additive(1/3),
                                                                    latent_state_increment = hcat(fill(rand(latent_state_distribution, length(content)), length(intermediate))...)
                                                                   ), ),
                              pre = intermediate,
                              post = content),
                   Connection(plasticity = (RewardModulatedHebbian(; pre = content,
                                                                     post = actuators,
                                                                     with_bias = with_readout_bias,
                                                                     η, η_bias, clipper,
                                                                     reward_sensor),),
                              pre = content, post = actuators,
                              with_bias = with_readout_bias,
                              modulator = OnlyActiveAtStep(3, stepper),
                              w = initial_weights(actuators, content, winit[1], winit[2]; only_first, with_bias = with_readout_bias)
                             )
                  )
    Brain(; reward_sensor, token_sensor, connections, actuators, is_micro_step! = stepper)
end


function run_sims(; brain, version = 1, n = 10^4)
    vcat([DataFrame(test_interval = i,
                    action = run_task1!(brain(),
                                       stimuli_task1(; test_interval = i, version)))
                for _ in 1:n, i = 1:5]...)
end
function analyse_results(results)
    combine(groupby(results, :test_interval),
            :action => (x -> mean(first.(x) .== 2)) => :act2,
            :action => (x -> mean(first.(filter(x -> x[2][1] == 1, x)) .== 2)) => :act2_given_posrew1,
            :action => (x -> mean(first.(filter(x -> x[2][1] == -1, x)) .== 2)) => :act2_given_negrew1,
            :action => (x -> mean(first.(last.(x)))) => :rew1,
            :action => (x -> mean(last.(last.(x)))) => :rew2,
           )
end
function action_prior(; winit = [.9, .2],
                        brain = increment_post_brain(; η = 0., winit),
                        n = 10^5)
    as = [step!.(Ref(brain), stimuli_task1(test_interval = rand(1:5)), 0)[end] for _ in 1:n]
    mean(as .== 2)
end

# TODO: What happens without Shifter? Results should remain the same.
learning_modes = Dict(:simple => (act = identity,),
                      :linear => (act = IdPlus(1.),),
                      :exp => (act = exp, clipper = Shifter()))
brains = Dict(:state_decay => hebbian_latent_state_decay_brain,
              :increment => increment_post_brain,
              :sparse => sparse_random_brain,
              :chrono => chrono_brain,
             )

push!(PGFPlotsX.CUSTOM_PREAMBLE, read(joinpath(DOCPATH, "modelnames.tex"), String))

function modelnames(s; extra = Dict())
    haskey(extra, s) && return extra[s]
    spl = split("$s", '_')
    name = if spl[1] == "sparse"
#         "Sparse-Random-Pruning"
        raw"\sparse{}"
    elseif spl[1] == "increment"
#         "Representational-Drift"
        raw"\incrementmodel{}"
    elseif spl[1] == "state"
#         "Organized-Pruning"
        raw"\statedecay{}"
    elseif spl[1] == "chrono"
#         "Chronological-Organization"
        raw"\chrono{}"
    end
    name # * " ($(spl[end]))"
end
function styles(s)
    spl = split("$s", '_')
    color = if spl[1] == "sparse"
        colors[5]
    elseif spl[1] == "increment"
        colors[3]
    elseif spl[1] == "state"
        colors[1]
    elseif spl[1] == "chrono"
        colors[4]
    else
        "black"
    end
    linestyle = if spl[end] == "exp"
        "solid"
    elseif spl[end] == "simple"
        "solid"
    elseif spl[end] == "linear"
        "dotted"
    else
        "solid"
    end
    st = @pgf {color = color}
    st[linestyle] = nothing
    st
end


###
### TASK 1
###

"""
These learning rates have approximately the best final performance in task2 and task3
among `η in (.005, .01, .02, .05, .1, .2, .5, 1., 2.)`.
"""
function learning_rate(b)
    b == :chrono && return 1.0
    b == :increment && return 0.5
    b == :sparse && return .02
    b == :state_decay && return .2
    error()
end

res1 = Dict()

for (nb, b) in brains
    for (nl, l) in learning_modes
        id = Symbol(nb, "_", nl)
#         nb == :sparse || continue
        nl == :exp || continue
        winit = [.5, .5]
        η = learning_rate(nb)
        @show id
        brain = () -> b(; η, winit, l...)
        res1[id] = run_sims(; brain)
    end
end

res1[:ap] = action_prior(winit = [.5, .5])

csave(joinpath(DATAPATH, "task1.dat.zstd"), res1)
res1 = cload(joinpath(DATAPATH, "task1.dat.zstd"))

ks = sort(collect(filter(x -> x != :ap && (false || split("$x", '_')[end] == "exp"), keys(res1))))

f1 = @pgf PGFPlotsX.Axis({xlabel = raw"$\Delta t_\mathrm{test}$",
           ymin = .4, ymax = .75, xtick = 1:5,
           xmin = .7, xmax = 5.3,
           legend_pos = "outer north east",
           ylabel = raw"probability of action $a_2$"},
           [Plot({mark = "*", styles(k)...},
                 Coordinates(1:5, analyse_results(res1[k]).act2))
            for k in ks]...,
           Plot({black, dotted}, Expression("$(res1[:ap])")),
           PGFPlotsX.Legend([modelnames.(ks);
                             "before learning"])
          )

pgfsave(joinpath(DOCPATH, "sim1.tikz"), f1)


###
### TASK 2
###

res2 = Dict()
res2[:optimal] = mean([run_task2!(OptimalLearner(), stimuli_task2(trials = 400), Tracker1()).r for _ in 1:10^2])


for (nb, b) in brains
    for (nl, l) in learning_modes
        nl == :exp || continue
        id = Symbol(nb, "_", nl)
        η = learning_rate(nb)
        N = 10^2
        @show id
        res2[id] = mean([run_task2!(b(; η, winit = [.5, .5], l...), stimuli_task2(trials = 400), Tracker1()).r for _ in 1:N])
    end
end

csave(joinpath(DATAPATH, "task2.dat.zstd"), res2)
res2 = cload(joinpath(DATAPATH, "task2.dat.zstd"))

session_average(x, l = 4) = [mean(x[(i-1)*l+1:i*l]) for i in 1:length(x)÷l]

ks = sort(collect(filter(x -> x != :optimal && split("$x", '_')[end] == "exp", keys(res2))))

# ks = best_of_kind.(Ref(res2), ["increment_exp", "sparse_exp", "state_decay_exp", "chrono_exp"], session_average) |> sort

f2 = @pgf PGFPlotsX.Axis({legend_pos = "outer north east", xlabel = "session",
           ylabel = "expected reward"},
          [Plot({styles(k)...},
                Coordinates(1:200, session_average(res2[k])))
            for k in ks]...,
           Plot({dotted}, Coordinates(1:200, session_average(res2[:optimal]))),
           Plot({dotted, orange}, Coordinates(1:200, fill(0.5, 200))),
           PGFPlotsX.Legend([modelnames.(ks); "optimal"; "best linear"
                            ])
          )

pgfsave(joinpath(DOCPATH, "sim2.tikz"), f2)


###
### TASK 3
###

res3 = Dict()
res3[:optimal] = mean([run_task3!(OptimalLearner(), stimuli_task3(trials = 100), Tracker1()).r for _ in 1:10^4])

for (nb, b) in brains
    for (nl, l) in learning_modes
            id = Symbol(nb, "_", nl)
            nl == :exp || continue
            η = learning_rate(nb)
            N = 10^3
            @show id
            res3[id] = mean([run_task3!(b(; η, winit = [.5, .5], l...), stimuli_task3(trials = 100), Tracker1()).r for _ in 1:N])
    end
end

csave(joinpath(DATAPATH, "task3.dat.zstd"), res3)
res3 = cload(joinpath(DATAPATH, "task3.dat.zstd"))

function best_of_kind(res, key, f = identity)
    ks = filter(x -> match(Regex(key), String(x)) !== nothing, keys(res))
    argmax(k -> f(res[k])[end], ks)
end

ks = sort(collect(filter(x -> x != :optimal && split("$x", '_')[end] == "exp", keys(res3))))

f3 = @pgf PGFPlotsX.Axis({legend_pos = "outer north east", xlabel = "trial",
               ylabel = "expected reward"},
               [Plot({styles(k)...},
                     Coordinates(1:100, res3[k])) for k in ks]...,
               Plot({dotted}, Coordinates(1:100, res3[:optimal])),
               PGFPlotsX.Legend([modelnames.(ks); "optimal"])
              )


pgfsave(joinpath(DOCPATH, "sim3.tikz"), f3)


###
### Food Caching Simulation
###

function stimuli_foodcaching(; T = 20_000, n_color = 10_000, n_shape = 10)
    sequence = Token{Int,Int}[]
    locations = randperm(n_color-1)
    cache_event_counter = 1
    for t in 1:T
        if length(sequence) > 10 && rand() > 1/3 # probe
            # power law sampling of past locations
            recall_weight = 1 ./ sqrt.((length(sequence):-1:1))
            c = 0
            while true
                idx = NeuralProcessingOfTime.wsample(recall_weight)
                token = sequence[idx]
                if token.s > 0 # look only at cache locations
                    c = token.c
                    break
                end
            end
            token = Token(c, 0)
        else
            token = Token(locations[cache_event_counter], rand(1:n_shape-1))
            cache_event_counter += 1
        end
        push!(sequence, token)
    end
    sequence
end
function optimal_action_foodcaching(history, query)
    query.s != 0 && return 2
    age = age_at_similarity(history, query, threshold = 1, similarity = NeuralProcessingOfTime.csimilarity)
    token = history[end-age+1]
    2-Int(age < (token.s == 0 ? 0 : token.s == 1 ? 10. : token.s == 2 ? 20. : 160.))
end
function optimal_action_oneage_foodcaching(history, query, refage)
    query.s != 0 && return 2
    age = age_at_similarity(history, query, threshold = 1, similarity = NeuralProcessingOfTime.csimilarity)
    token = history[end-age+1]
    2-Int(age < (token.s == 0 ? 0 : refage))
end
function reward_foodcaching(history, query, action)
    (query.s != 0 || action == 0 || action == 2) && return 0.
    age = age_at_similarity(history, query, threshold = 1, similarity = NeuralProcessingOfTime.csimilarity)
    token = history[end-age+1]
    float(age < (token.s == 0 ? 0 : token.s == 1 ? 10. : token.s == 2 ? 20. : 160.))-.2
end
function foodcaching_rewards(stimuli, policy)
    [reward_foodcaching(stimuli[1:t-1], stimuli[t],
                    policy(stimuli[1:t-1], stimuli[t]))
     for t in eachindex(stimuli)]
end
function special_connections!(bigbrain; n_color, n_shape, only_shape_content = false, ks = [10, 20, 160])
    for i in 1:n_color-1, j in 1:2
        bigbrain.connections[1].idxs[2*(i-1) + j] = [i]
    end
    if only_shape_content
        for j in 1:n_shape-1, k in eachindex(ks)
            post_idx = length(ks)*(j-1) + k
            bigbrain.connections[2].idxs[post_idx] = [n_color-1 + j]
            bigbrain.connections[3].plasticity[1].latent_state_increment[post_idx, :] .= ks[k]
            bigbrain.connections[4].w[1, post_idx] = j == 1 ? .4 : j == 2 ? .9 : 1.2
            bigbrain.connections[4].w[2, post_idx] = 0
        end
    else
        for i in 1:n_color-1, j in 1:n_shape-1, k in eachindex(ks)
            post_idx = length(ks)*(i-1) + length(ks)*(n_color-1)*(j-1) + k
            bigbrain.connections[2].idxs[post_idx] = [i, n_color-1 + j]
            bigbrain.connections[3].plasticity[1].latent_state_increment[post_idx, :] .= ks[k]
            bigbrain.connections[4].w[1, post_idx] = j == 1 ? .4 : j == 2 ? .9 : 1.2
            bigbrain.connections[4].w[2, post_idx] = 0
        end
    end
    # bias
    bigbrain.connections[4].w[1, end] = 0.
    bigbrain.connections[4].w[2, end] = 1.
    bigbrain
end

Random.seed!(1221)
n_color = 1000
n_shape = 4
T = 2*n_color
bigbrain = sparse_random_brain(; act = exp, clipper = identity, n_intermediate = 5*(n_color-1), n_content = 20*(n_shape-1), latent_state_increment_factor = 200, n_color, n_shape, η = .02, η_bias = 0, winit = [1/10, 0], max_fanin = 3, min_fanin = 1, sparsity = 1/n_color, in_weights = [fill(1/(n_color-1), n_color - 1); fill(2/(n_shape-1), n_shape - 1)], content_activation = heaviside)
bigbrain.connections[4].w[:, end] .= [0, 1]
results = []
baseline_policies = (best = optimal_action_foodcaching,
                     worst = (h, ht) -> optimal_action_foodcaching(h, ht) == 1 ? 2 : 1,
                     random = (_, _) -> rand(1:2),
                     all1 = (_, _) -> 1,
                     all2 = (_, _) -> 2,
                     oneage1 = (h, ht) -> optimal_action_oneage_foodcaching(h, ht, 10),
                     oneage2 = (h, ht) -> optimal_action_oneage_foodcaching(h, ht, 20),
                     oneage3 = (h, ht) -> optimal_action_oneage_foodcaching(h, ht, 160))
Random.seed!(20)
for epoch in 1:30
    stimuli = stimuli_foodcaching(; T, n_color, n_shape);
    baselines = map(p -> foodcaching_rewards(stimuli, p), baseline_policies)
    @time res = run_task!(bigbrain, stimuli, reward_foodcaching, TrackAll(), greedy = true)
    @show epoch map(sum, baselines) sum(res.r)
    push!(results, (; stimuli, baselines, res))
end

csave(joinpath(DATAPATH, "foodcaching.dat.zstd"), results)

Random.seed!(1221)
n_color = 1000
n_shape = 4
T = 2*n_color
bigbrain = sparse_random_brain(; act = exp, clipper = identity, n_intermediate = 2*(n_color-1), n_content = 3*(n_shape-1), latent_state_increment_factor = 200, n_color, n_shape, η = .02, η_bias = 0., winit = [.5, .5], max_fanin = 3, min_fanin = 2, sparsity = 2/n_color, in_weights = [fill(1/(n_color-1), n_color - 1); fill(2/(n_shape-1), n_shape - 1)], content_activation = heaviside)
special_connections!(bigbrain; n_color, n_shape, only_shape_content = true)
bigbrain.connections[4].w[1, 1:end-1] .= 1.01
bigbrain.connections[4].w[2, 1:end-1] .= 0
results2 = []
Random.seed!(20)
for epoch in 1:30
    stimuli = stimuli_bigtask(; T, n_color, n_shape);
    baselines = map(p -> bigtask_rewards(stimuli, p), baseline_policies)
    @time res = run_task!(bigbrain, stimuli, reward_bigtask, TrackAll(), greedy = true)
    @show epoch map(sum, baselines) sum(res.r)
    push!(results2, (; stimuli, baselines, res))
end

csave(joinpath(DATAPATH, "foodcaching2.dat.zstd"), results2)

function result_dataframe(results, baseline_policies)
    resultdf = vcat([DataFrame(map(p -> foodcaching_rewards(r.stimuli, p),
                                   baseline_policies))
                     for r in results]...)
    resultdf.model = vcat(getproperty.(getproperty.(results, :res), :r)...)
    test = vcat([[t.s == 0 for t in r.stimuli] for r in results]...)
    resultdf = resultdf[test, :]
end
resultdf1 = result_dataframe(results, baseline_policies)
resultdf2 = result_dataframe(results2, baseline_policies)

struct Smoother
    weights::Vector{Float64}
    center::Int
    stride::Int
end
Smoother(; σ = 500, b = 8, stride = σ÷10) = Smoother([exp(-d^2/σ^2) for d in -b*σ:b*σ], b * σ, stride)
(s::Smoother)(x) = s.(Ref(x), 1:s.stride:length(x))
function (s::Smoother)(x, i)
    n = 0.
    y = 0.
    ws, c = s.weights, s.center
    for j in max(1, i - c):min(length(x), i + c)
        w = ws[j - i + c + 1]
        y += w * x[j]
        n += w
    end
    y/n
end
struct Scaler
    min::Vector{Float64}
    max::Vector{Float64}
end
(s::Scaler)(x) = @. (x - s.min) / (s.max - s.min)

smoother = Smoother(σ = 500)
scaler = Scaler(smoother(resultdf2.random), smoother(resultdf2.best))

ks = setdiff(names(resultdf1), ["worst", "all2"])
modelname = Dict("best" => "optimal",
                 "random" => "uniform random",
                 "all1" => "always retrieve",
                 "all2" => "never retrieve",
                 "oneage1" => "retrieve if age \$\\leq\$ 10 steps",
                 "oneage2" => "retrieve if age \$\\leq\$ 20 steps",
                 "oneage3" => "retrieve if age \$\\leq\$ 160 steps",
                 "model" => modelnames("chrono"),
                 "model2" => modelnames("sparse"))
cs = Dict("best" => @pgf({black, dotted, thick}),
          "worst" => @pgf({black, dotted}),
          "random" => @pgf({black,dashed}),
          "all1" => @pgf({green!50!black}),
          "all2" => @pgf({blue,dashed}),
          "oneage1" => @pgf({green!50!black,dashed}),
          "oneage2" => @pgf({green!50!black,"dash dot"}),
          "oneage3" => @pgf({green!50!black,dotted}),
          "model" => styles("chrono"),
          "model2" => styles("sparse"),
         )

f4 = @pgf PGFPlotsX.Axis({legend_pos = "outer north east", xlabel = "retrieval test event",
               ylabel = "average normalized reward", ymin = -.1, "very thick"},
            Plot(cs["model2"], Coordinates(1:smoother.stride:nrow(resultdf1),
                                          scaler(smoother(resultdf1.model)))),
            [Plot(cs[k],
                  Coordinates(1:smoother.stride:nrow(resultdf1), scaler(smoother(getproperty(resultdf2, k))))) for k in reverse(ks)]...,
            PGFPlotsX.Legend(map(x -> modelname[x], reverse([ks; "model2"])))
              )


pgfsave(joinpath(DOCPATH, "sim_caching.tikz"), f4)

