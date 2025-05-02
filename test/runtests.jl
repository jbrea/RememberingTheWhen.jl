using NeuralProcessingOfTime, Test
import NeuralProcessingOfTime: Token, Color, Shape, age_at_similarity
import NeuralProcessingOfTime: Neurons, Distributed, Connection, TokenSensor, Hebbian,
       One2OneConnection, All2AllConnection, HebbianLatentStateDecay, Brain, RewardSensor,
       RewardModulatedHebbian, FixedMicroSteps
import NeuralProcessingOfTime: sense!, micro_step!, update!, propagate!, activity,
       setactivity!, heaviside, step!, action!
import NeuralProcessingOfTime: blue, red, black, square, triangle, yellow, shapeless

@testset "stimuli" begin
    history = [Token(red, triangle), Token(blue, triangle), Token(red, square)]
    @test age_at_similarity(history, Token(red, square)) == 1
    @test age_at_similarity(history, Token(red, triangle)) == 3
    @test age_at_similarity(history, Token(black, triangle), threshold = 0.5) == 2
    @test age_at_similarity(history, Token(black, triangle)) === nothing
end

@testset "basic input-output" begin
    n_colors = length(instances(Color))-1
    actuators = Neurons("actuators", Distributed(x = zeros(2)))
    token_sensor = TokenSensor()
    hidden = Neurons("hidden", Distributed(length(token_sensor.neurons)))
    connections = (
                   One2OneConnection(pre = token_sensor.neurons,
                                     post = hidden),
                   Connection(plasticity = (Hebbian(), ),
                              pre = hidden,
                              post = actuators),
                  )
    sense!(token_sensor, Token(red, triangle))
#     @test token_sensor.neurons.code.x.i == Int(red)
#     @test token_sensor.neurons.code.y.i == Int(triangle)
#     sense!(token_sensor, Token(yellow, square))
#     @test token_sensor.neurons.code.x.i == Int(yellow)
#     @test token_sensor.neurons.code.y.i == Int(square)
    brain = Brain(; token_sensor, connections, actuators)
    sense!(token_sensor, Token(blue, square))
#     micro_step!(brain)
    @test token_sensor.neurons.code.x.i₋ == Int(blue)
    @test token_sensor.neurons.code.y.i₋ == Int(square)
    @test activity(token_sensor.neurons.code.x, Int(blue), previous = true) == 1
    @test activity(token_sensor.neurons, Int(blue), previous = true) == 1
    @test activity(token_sensor.neurons, n_colors+Int(square), previous = true) == 1
    @test activity(token_sensor.neurons.code.x, Int(blue)) == 0
    setactivity!(actuators, 1, 1.)
    for connection in brain.connections
        propagate!(connection)
    end
    for connections in brain.connections
        update!(connections)
    end
    @test hidden.code.x[Int(blue)] == 1
    @test hidden.code.x[Int(square) + n_colors] == 1
    for neurons in brain.neurons
        update!(neurons)
    end
    @test hidden.code.x₋[Int(blue)] == 1
    @test hidden.code.x₋[Int(square) + n_colors] == 1
    @test sum(connections[2].w[1, :]) == 2
    @test sum(connections[2].w[2, :]) == 0
end

@testset "storage recall" begin
    actuators = Neurons("actuators", Distributed(x = zeros(2)))
    token_sensor = TokenSensor()
    intermediate = Neurons("intermediate", Distributed(length(token_sensor.neurons)))
    content = Neurons("content", Distributed(length(token_sensor.neurons)))
    tag = Neurons("tag", Distributed(6), heaviside)
    connections = (
                   One2OneConnection(pre = token_sensor.neurons,
                                     post = intermediate),
                   One2OneConnection(pre = token_sensor.neurons,
                                     post = content),
                   All2AllConnection(pre = token_sensor.neurons,
                                     post = tag),
                   Connection(plasticity = (Hebbian(), ),
                              pre = intermediate,
                              post = content),
                   Connection(plasticity = (HebbianLatentStateDecay(pre = intermediate, post = tag), ),
                              pre = intermediate,
                              post = tag),
                  )
    brain = Brain(; token_sensor, connections, actuators)
    sense!(token_sensor, Token(red, triangle))
    micro_step!(brain)
    @test sum(connections[4].w) == 4
    @test maximum(connections[5].plasticity[1].latent_state) == 5
    micro_step!(brain) # effective weight update is delayed
    @test sum(connections[5].w) == 10
    micro_step!(brain) # effective weight update is delayed
    @test sum(connections[5].w) == 8
end

@testset "reward modulated plasticity" begin
    actuators = Neurons("actuators", Distributed(x = zeros(2)))
    token_sensor = TokenSensor()
    reward_sensor = RewardSensor()
    connections = (Connection(plasticity = (RewardModulatedHebbian(; pre = token_sensor.neurons,
                                                                     post = actuators,
                                                                     reward_sensor),),
                              pre = token_sensor.neurons, post = actuators,
                              w = fill(.1, length(actuators), length(token_sensor.neurons))),
                  )
    brain = Brain(; token_sensor, reward_sensor, connections, actuators,
                    is_micro_step! = FixedMicroSteps(N = 1))
    sense!(brain.token_sensor, Token(red, triangle))
    sense!(brain.reward_sensor, 0.)
    propagate!(brain.connections)
    a1 = action!(brain.actuators)
    @test activity(actuators, a1) == 1
    update!(brain.connections)
    @test connections[1].plasticity[1].eligibility[a1, 1] == 1
    update!(brain.neurons)
    a2 = step!(brain, Token(red, triangle), 1.)
    @test connections[1].w[a1, 1] == 1.1
    @test connections[1].w[a1 % 2 + 1, 1] == .1
    a3 = step!(brain, Token(red, triangle), -.3)
    if a1 == a2
        @test connections[1].w[a1, 1] == .8
        @test connections[1].w[a1 % 2 + 1, 1] == .1
    else
        @test connections[1].w[a1, 1] == 1.1
        @test connections[1].w[a1 % 2 + 1, 1] == -.1
    end
end
