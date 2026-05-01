This repository contains code for the simulations reported in the article

Remembering the "When": Hebbian Memory Models for the Time of Past Events, Johanni Brea, Alireza Modirshanechi, Georgios Iatropoulos, Wulfram Gerstner.

To run the simulations, clone the repository, navigate to the `scripts` folder and run the following code in a Julia REPL (tested with Julia version 1.9).

```julia
using Pkg
Pkg.activate()
Pkg.instantiate()

mkdir(joinpath(@__DIR__, "..", "doc"))
mkdir(joinpath(@__DIR__, "..", "data"))

include("sims.jl")
```

## Experimental Protocols
In the code the experimental protocols are written in a compact form. For readability we provide the relevant parts here explicitly.

### Generalization of actions (Fig. 9)
```julia
results = DataFrame(a2_test = [])
for subject in subjects
    show_to!(subject, red_triangle)
    show_to!(subject, blue_square)
    show_to!(subject, green_circle)
    show_to!(subject, red_triangle)
    a_train = force_to_choose_and_observe_action!(subject)
    if a_train == a_1
        reward!(subject, -1)
    else
        reward!(subject, +1)
    end
    delta_test = rand([1, 2, 3, 4, 5])
    if delta_test == 1
        show_to!(subject, yellow_pentagon)
        show_to!(subject, yellow_pentagon)
    elseif delta_test == 2
        show_to!(subject, green_circle)
    elseif delta_test == 3
        show_to!(subject, blue_square)
    elseif delta_test == 4
        show_to!(subject, yellow_pentagon)
        show_to!(subject, blue_square)
    else
        show_to!(subject, yellow_pentagon)
        show_to!(subject, orange_halfcircle)
        show_to!(subject, blue_square)
    end
    a_test = force_to_choose_and_observe_action!(subject)
    push!(results, is_equal(a_test, a2))
end
# compute average number of a2 responses
```

### Learning to take age-dependent actions (Fig. 10)
```julia
results = DataFrame(trial = [], reward = [])
for subject in subjects
    for trial in trials
        show_to!(subject, red_triangle)
        delta = rand([1, 2, 3, 4, 5])
        for i in 1:delta-1
            show_to!(subject, black_circle)
        end
        show_to!(subject, red_triangle)
        a = force_to_choose_and_observe_action!(subject)
        if (delta <= 2 && a == a_2) || (delta > 2 && a == a_1)
            reward!(subject, +1)
            push!(results, [trial, +1])
        else
            reward!(subject, -1)
            push!(results, [trial, -1])
        end
        # do something else before new trial starts
    end
end
```

### Learning to take age- and content-dependent actions (Fig. 11)
```julia
results = DataFrame(session = [], reward = [])
for subject in subjects
    for session in sessions
        r_per_session = 0
        # trial 1
        show_to!(subject, red_triangle)
        show_to!(subject, black_circle)
        show_to!(subject, red_triangle)
        a = force_to_choose_and_observe_action!(subject)
        if a == a_1
            reward!(subject, -1)
            r_per_session -= 1
        else
            reward!(subject, +1)
            r_per_session += 1
        end
        # do something else before trial 2 starts
        show_to!(subject, red_triangle)
        show_to!(subject, black_circle)
        show_to!(subject, black_circle)
        show_to!(subject, red_triangle)
        a = force_to_choose_and_observe_action!(subject)
        if a == a_1
            reward!(subject, +1)
            r_per_session += 1
        else
            reward!(subject, -1)
            r_per_session -= 1
        end
        # do something else before trial 3 starts
        show_to!(subject, blue_square)
        show_to!(subject, black_circle)
        show_to!(subject, blue_square)
        a = force_to_choose_and_observe_action!(subject)
        if a == a_1
            reward!(subject, +1)
            r_per_session += 1
        else
            reward!(subject, -1)
            r_per_session -= 1
        end
        # do something else before trial 4 starts
        show_to!(subject, blue_square)
        show_to!(subject, black_circle)
        show_to!(subject, black_circle)
        show_to!(subject, blue_square)
        a = force_to_choose_and_observe_action!(subject)
        if a == a_1
            reward!(subject, -1)
            r_per_session -= 1
        else
            reward!(subject, +1)
            r_per_session += 1
        end
        push!(results, [session, r_per_session/4])
        # do something else before new session starts
    end
end
# compute average reward per session
```
