module RNNLM
using Knet
using AutoGrad
using ArgParse
using ReverseDiff
using ReverseDiff: gradient, GradientTape, compile, gradient!


const train_file = "data/text/train.txt"
const test_file = "data/text/dev.txt"
const SOS = "<s>"
t00 = now()

function main(args=ARGS)
    s = ArgParseSettings()
    s.description = "RNN Language Model in Knet"
    s.exc_handler=ArgParse.debug_handler

    @add_arg_table s begin
        ("--gpu"; action=:store_true; help="use GPU or not")
        ("MB_SIZE"; arg_type=Int; help="minibatch_size")
        ("EMBED_SIZE"; arg_type=Int; help="embedding size")
        ("HIDDEN_SIZE"; arg_type=Int; help="hidden size")
        ("SPARSE"; arg_type=Int; help="sparse update 0/1")
        ("TIMEOUT"; arg_type=Int; help="max timeout")
        ("--train"; default=train_file; help="train file")
        ("--test"; default=test_file; help="test file")
        ("--seed"; arg_type=Int; default=-1; help="random seed")
        ("--epochs"; arg_type=Int; default=100; help="epochs")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:seed] > 0 && srand(o[:seed])
    atype = o[:gpu] ? KnetArray{Float32} : Array{Float32}

    # build data
    w2i = Dict()
    trn = read_data(o[:train], w2i)
    tst = read_data(o[:test], w2i)
    sort!(trn, by=length, rev=true)
    sort!(tst, by=length, rev=true)
    trn, tst = map(split->make_batches(split, w2i, o[:MB_SIZE]), [trn, tst])

    # build model
    w = initweights(atype, o[:HIDDEN_SIZE], length(w2i), o[:EMBED_SIZE])
    s0 = initstate(atype, o[:HIDDEN_SIZE], o[:MB_SIZE])
    s1 = initstate(atype, o[:HIDDEN_SIZE], size(trn[end][1][1],1))
    s2 = initstate(atype, o[:HIDDEN_SIZE], size(tst[end][1][1],1))
    opt = map(x->Adam(), w)

    # compile tape
    f_tape = GradientTape(loss, (w...,s0...,trn[1][1]...))
    compiled = compile(f_tape)

    # train language model
    println("startup time: ", Int(now()-t00)*0.001); flush(STDOUT)
    t0 = now()
    all_time = dev_time = all_tagged = this_words = this_loss = 0
    for epoch = 1:o[:epochs]
        shuffle!(trn)
        for k = 1:length(trn)
            iter = (epoch-1)*length(trn) + k
            if iter % div(500, o[:MB_SIZE]) == 0
                @printf("%f\n", this_loss/this_words); flush(STDOUT)
                all_tagged += this_words
                this_loss = this_words = 0
                all_time = Int(now()-t0)*0.001
            end

            if iter % div(10000, o[:MB_SIZE]) == 0
                dev_start = now()
                dev_loss = dev_words = 0
                for i = 1:length(tst)
                    seq, nwords = tst[i]
                    s = o[:MB_SIZE] == size(seq[1],1) ? s0 : s2
                    dev_loss += loss(w...,s...,seq...)
                    dev_words += nwords
                end
                dev_time += Int(now()-dev_start)*0.001
                train_time = Int(now()-t0)*0.001-dev_time

                @printf(
                    "nll=%.4f, ppl=%.4f, words=%d, time=%.4f, word_per_sec=%.4f\n",
                    dev_loss/dev_words, exp(dev_loss/dev_words), dev_words,
                    train_time, all_tagged/train_time); flush(STDOUT)

                if all_time > o[:TIMEOUT]
                    return
                end
            end

            # train on minibatch
            seq, batch_words = trn[k]
            s = o[:MB_SIZE] == size(seq[1],1) ? s0 : s1
            batch_loss = train!(w,s,seq,opt,compiled)
            this_loss += batch_loss
            this_words += batch_words
        end
        @printf("epoch %d finished\n", epoch-1); flush(STDOUT)
    end
end

# build vocabulary, training and test data
function read_data(file, w2i)
    get_tokens(line) = [split(line, " ")[2:end-1]; SOS]
    data = open(file, "r") do f
        data = []
        for ln in readlines(f)
            words = get_tokens(ln)
            senvec = []
            for word in words
                if !haskey(w2i, word)
                    w2i[word] = length(w2i)+1
                end
                push!(senvec, w2i[word])
            end
            push!(data, senvec)
        end
        data
    end
end

# make minibatches
function make_batches(data, w2i, batchsize)
    batches = []
    for k = 1:batchsize:length(data)
        samples = data[k:min(k+batchsize-1, length(data))]
        lengths = map(length, samples)
        longest = reduce(max, lengths)
        nwords = sum(lengths)
        nsamples = length(samples)
        pad = length(w2i)
        seq = map(i -> pad * ones(Int, nsamples), [1:longest...])
        for i = 1:nsamples
            map!(t->seq[t][i] = samples[i][t], [1:length(samples[i])...])
        end
        push!(batches, (seq, nwords))
    end
    return batches
end

# initialize hidden and cell arrays
function initstate(atype, hidden, batchsize)
    state = Array(Any, 2)
    state[1] = zeros(hidden, batchsize)
    state[2] = zeros(hidden, batchsize)
    return map(s->convert(atype,s), state)
end

# initialize all weights of the language model
# w[1:2] => weight/bias params for LSTM network
# w[3:4] => weight/bias params for softmax layer
# w[5]   => word embeddings
function initweights(atype, hidden, vocab, embed, winit=0.01)
    w = Array(Any, 5)
    input = embed
    w[1] = winit*randn(4*hidden, hidden+input)
    w[2] = zeros(4*hidden, 1)
    w[2][1:hidden] = 1 # forget gate bias
    w[3] = winit*randn(vocab+1, hidden)
    w[4] = zeros(vocab+1, 1)
    w[5] = winit*randn(embed, vocab+1)
    return map(i->convert(atype, i), w)
end

# LSTM model -i nput * weight, concatenated weights
function lstm(weight, bias, hidden, cell, input)
    gates   = weight * vcat(hidden,input) .+ bias
    hsize   = size(hidden,1)
    forget  = sigm(gates[1:hsize,:])
    ingate  = sigm(gates[1+hsize:2hsize,:])
    outgate = sigm(gates[1+2hsize:3hsize,:])
    change  = tanh(gates[1+3hsize:end,:])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden,cell)
end

# LSTM prediction
function predict(w, s, x)
    emb = w[5][:,x] # size(w[5]): ExV
    (s[1],s[2]) = lstm(w[1],w[2],s[1],s[2],emb)
    return w[3] * s[1] .+ w[4]
end

function logprob(output, ypred)
    nrows,ncols = size(ypred)
    index = output + nrows*(0:(length(output)-1))
    o1 = logp(ypred,1)
    o2 = o1[index]
    o3 = sum(o2)
    return o3
end

# LM loss function
function loss(params...)
    w = params[1:5]
    s = params[6:7]
    seq = params[8:end]
    total = 0
    input = seq[1]
    for t in 1:length(seq)-1
        ypred = predict(w, s, input)
        ygold = seq[t+1]
        total += logprob(ygold,ypred)
        input = ygold
    end

    # push!(values, AutoGrad.getval(-total))
    return -total
end

# lossgradient = grad(loss)

function train!(w,s,seq,opt,compiled)
    params = (w...,s...,seq...)
    gradient!(g,compiled,params)
    update!(w,g,opt)
end

if VERSION >= v"0.5.0-dev+7720"
    PROGRAM_FILE=="knet/rnnlm-batch.jl" && main(ARGS)
else
    !isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)
end

end # module
