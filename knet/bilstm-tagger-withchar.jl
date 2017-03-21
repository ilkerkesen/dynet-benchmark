module CharTagger
using Knet
using AutoGrad
using ArgParse

const train_file = "data/tags/train.txt"
const test_file = "data/tags/dev.txt"
const UNK = "_UNK_"
const PAD = "<*>"
t00 = now()

function main(args)
    s = ArgParseSettings()
    s.description = "Bidirectional LSTM Tagger (with chars) in Knet"

    @add_arg_table s begin
        ("--gpu"; action=:store_true; help="use GPU or not")
        ("CEMBED_SIZE"; arg_type=Int; help="char embedding size")
        ("WEMBED_SIZE"; arg_type=Int; help="embedding size")
        ("HIDDEN_SIZE"; arg_type=Int; help="hidden size")
        ("MLP_SIZE"; arg_type=Int; help="MLP size")
        ("SPARSE"; arg_type=Int; help="sparse update 0/1")
        ("TIMEOUT"; arg_type=Int; help="max timeout")
        ("--batchsize"; arg_type=Int; help="minibatch size"; default=1)
        ("--train"; default=train_file; help="train file")
        ("--test"; default=test_file; help="test file")
        ("--seed"; arg_type=Int; default=-1; help="random seed")
        ("--epochs"; arg_type=Int; default=100; help="epochs")
        ("--minoccur"; arg_type=Int; default=6)
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:seed] > 0 && srand(o[:seed])
    atype = o[:gpu] ? KnetArray{Float32} : Array{Float32}

    # read data
    trn = read_file(o[:train])
    tst = read_file(o[:test])

    # get words and tags from train set
    words, tags, chars = [], [], Set()
    for sample in trn
        for (word,tag) in sample
            word = String(word)
            push!(words, word)
            push!(tags, tag)
            push!(chars, convert(Array{UInt8,1}, word)...)
        end
    end
    chars = collect(chars)

    # count words and build vocabulary
    wordcounts = count_words(words)
    nwords = length(wordcounts) + 1 # for UNK
    wordcounts = filter((w,c)-> c >= o[:minoccur], wordcounts)
    words = collect(keys(wordcounts))
    push!(chars, PAD)
    w2i, i2w = build_vocabulary(words)
    t2i, i2t = build_vocabulary(tags)
    c2i, i2c = build_vocabulary(chars)
    ntags = length(t2i)
    nchars = length(c2i)
    !haskey(c2i, PAD) && error("...")

    # build model
    w = initweights(atype, o[:HIDDEN_SIZE], length(w2i), ntags, nchars,
                    o[:WEMBED_SIZE], o[:CEMBED_SIZE], o[:MLP_SIZE])
    s = initstate(atype, o[:HIDDEN_SIZE], o[:WEMBED_SIZE])
    opt = map(x->Adam(), w)

    # train bilstm tagger
    println("nwords=$nwords, ntags=$ntags, nchars=$nchars"); flush(STDOUT)
    println("startup time: ", Int(now()-t00)*0.001); flush(STDOUT)
    t0 = now()
    all_time = dev_time = all_tagged = this_tagged = this_loss = 0
    for epoch = 1:o[:epochs]
        shuffle!(trn)
        for k = 1:length(trn)
            iter = (epoch-1)*length(trn) + k
            if iter % 500 == 0
                @printf("%f\n", this_loss/this_tagged); flush(STDOUT)
                all_tagged += this_tagged
                this_loss = this_tagged = 0
                all_time = Int(now()-t0)*0.001
            end

            if iter % 10000 == 0 || all_time > o[:TIMEOUT]
                dev_start = now()
                good_sent = bad_sent = good = bad = 0.0
                for sample in tst
                    seq, is_word = make_input(sample, w2i, c2i)
                    ypred = predict(w, copy(s), seq, is_word)
                    ypred = map(x->i2t[x], ypred)
                    ygold = map(x -> x[2], sample)
                    same = true
                    for (y1,y2) in zip(ypred, ygold)
                        if y1 == y2
                            good += 1
                        else
                            bad += 1
                            same = false
                        end
                    end
                    if same
                        good_sent += 1
                    else
                        bad_sent += 1
                    end
                end
                dev_time += Int(now()-dev_start)*0.001
                train_time = Int(now()-t0)*0.001-dev_time

                @printf(
                    "tag_acc=%.4f, sent_acc=%.4f, time=%.4f, word_per_sec=%.4f\n",
                    good/(good+bad), good_sent/(good_sent+bad_sent), train_time,
                    all_tagged/train_time); flush(STDOUT)

                all_time > o[:TIMEOUT] && return
            end

            # train with instance
            seq, is_word = make_input(trn[k],w2i,c2i)
            out = make_output(trn[k],t2i)
            batch_loss = train!(w,s,seq,is_word,out,opt)
            this_loss += batch_loss
            this_tagged += length(trn[k])
        end
        @printf("epoch %d finished\n", epoch-1); flush(STDOUT)
    end
end

# parse line
function parse_line(line)
    return map(x->split(x,"|"), split(replace(line,"\n",""), " "))
end

# read file
function read_file(file)
    data = open(file, "r") do f
        map(parse_line, readlines(f))
    end
end

function count_words(words)
    wordcounts = Dict()
    for word in words
        wordcounts[word] = get(wordcounts, word, 0) + 1
    end
    return wordcounts
end

function build_vocabulary(words)
    words = collect(Set(words))
    w2i = Dict(); i2w = Dict()
    counter = 1
    for (i,word) in enumerate(words)
        w2i[word] = i
        i2w[i] = word
    end
    w2i, i2w
end

function make_input(sample, w2i, c2i)
    seq, is_word = Any[], Bool[]
    words = map(x->x[1], sample)
    for word in words
        push!(is_word, haskey(w2i, word))
        if is_word[end]
            push!(seq, w2i[word])
        else
            chars = [PAD; convert(Array{UInt8,1}, word); PAD]
            push!(seq, map(c->c2i[c], chars))
        end
    end
    (seq,is_word)
end

function make_output(sample,t2i)
    map(s->t2i[s[2]], sample)
end

# initialize hidden and cell arrays
function initstate(atype, hidden, wembed, batchsize=1)
    state = Array(Any, 4)
    state[1] = zeros(batchsize, hidden)
    state[2] = zeros(batchsize, hidden)
    state[3] = zeros(batchsize, div(wembed,2))
    state[4] = zeros(batchsize, div(wembed,2))
    return map(s->convert(atype,s), state)
end

# init LSTM parameters
function initlstm(input, hidden, winit=0.01)
    w = winit*randn(input+hidden, 4*hidden)
    b = zeros(1, 4*hidden)
    b[1:hidden] = 1
    return (w, b)
end

# initialize all weights of the language model
function initweights(
    atype, hidden, words, tags, chars, wembed, cembed, mlp, winit=0.01)
    w = Array(Any, 14)

    # LSTM params, 1:4 -> tagger LSTM, 5:8 -> char LSTM
    for k = 1:2:8
        a, b = initlstm(
            k < 4 ? wembed : cembed,
            k < 4 ? hidden : div(wembed, 2))
        w[k] = a; w[k+1] = b
    end

    # weight/bias params for MLP network
    w[9] = winit*randn(2*hidden, mlp)
    w[10] = zeros(1, mlp)
    w[11] = winit*randn(mlp, tags)
    w[12] = winit*randn(1, tags)

    # word/char embeddings
    w[13] = winit*randn(words, wembed)
    w[14] = winit*randn(chars, cembed)
    return map(i->convert(atype, i), w)
end

# LSTM model - input * weight, concatenated weights
function lstm(weight, bias, hidden, cell, input)
    gates   = hcat(input,hidden) * weight .+ bias
    hsize   = size(hidden,2)
    forget  = sigm(gates[:,1:hsize])
    ingate  = sigm(gates[:,1+hsize:2hsize])
    outgate = sigm(gates[:,1+2hsize:3hsize])
    change  = tanh(gates[:,1+3hsize:end])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden,cell)
end

# loss function
function loss(w, s, seq, is_word, out, values=[])
    # encoder
    sfs, sbs = encoder(w,s,seq,is_word)

    # prediction
    atype = typeof(AutoGrad.getval(w[1]))
    total = 0
    rng = 1:length(seq)
    for t in rng
        x = hcat(sfs[t], sbs[t]) * w[9] .+ w[10]
        ypred = x * w[11] .+ w[12]
        ygold = reshape(out[t:t], 1, 1)
        total += logprob(ygold,ypred)
    end

    push!(values, AutoGrad.getval(-total))
    return -total
end

# bilstm encoder
function encoder(w,s,seq,is_word)
    atype = typeof(AutoGrad.getval(w[1]))

    # embedding
    embed = Array(Any, length(seq))
    inds = convert(Array{Int32}, seq[is_word])
    wembed = w[end-1][inds,:]
    wi = 1

    for k = 1:length(seq)
        if is_word[k]
            embed[k] = wembed[wi:wi,:]; wi += 1; continue
        end

        # rare word embed
        inds = convert(Array{Int32}, seq[k])
        cembed = w[end][inds,:]
        rng = 1:length(inds)
        sf = copy(s[3:4])
        sb = copy(sf)
        for (ft,bt) in zip(rng,reverse(rng))
            (sf[1],sf[2]) = lstm(w[5],w[6],sf[1],sf[2],cembed[ft:ft,:])
            (sb[1],sb[2]) = lstm(w[7],w[8],sb[1],sb[2],cembed[bt:bt,:])
        end
        embed[k] = hcat(sf[1],sb[1])
    end

    # states and state histories
    sf, sb = copy(s[1:2]), copy(s[1:2])
    sfs = Array(Any, length(seq))
    sbs = Array(Any, length(seq))

    # encoding
    rng = 1:length(seq)
    for (ft,bt) in zip(rng,reverse(rng))
        # forward LSTM
        (sf[1],sf[2]) = lstm(w[1],w[2],sf[1],sf[2],embed[ft])
        sfs[ft] = copy(sf[1])

        # backward LSTM
        (sb[1],sb[2]) = lstm(w[3],w[4],sb[1],sb[2],embed[bt])
        sbs[bt] = copy(sb[1])
    end

    (sfs, sbs)
end

function logprob(output, ypred)
    nrows,ncols = size(ypred)
    index = similar(output)
    @inbounds for i=1:length(output)
        index[i] = i + (output[i]-1)*nrows
    end
    o1 = logp(ypred,2)
    o2 = o1[index]
    o3 = sum(o2)
    return o3
end

# tag given input sentence
function predict(w,s,seq,is_word)
    # encoder
    sfs, sbs = encoder(w,s,seq,is_word)

    # prediction
    tags = []
    atype = typeof(AutoGrad.getval(w[1]))
    rng = 1:length(seq)
    for t in rng
        x = hcat(sfs[t], sbs[t]) * w[9] .+ w[10]
        ypred = x * w[11] .+ w[12]
        ypred = convert(Array{Float32}, ypred)[:]
        push!(tags, indmax(ypred))
    end

    tags
end

lossgradient = grad(loss)

function train!(w,s,seq,is_word,out,opt)
    values = []
    gloss = lossgradient(w, copy(s), seq, is_word, out, values)
    for k = 1:length(w)
        gloss[k] != nothing && update!(w[k], gloss[k], opt[k])
    end
    isa(s,Vector{Any}) || error("State should not be Boxed.")
    for k = 1:length(s)
        s[k] = AutoGrad.getval(s[k])
    end
    values[1]
end

if VERSION >= v"0.5.0-dev+7720"
    PROGRAM_FILE=="knet/bilstm-tagger-withchar.jl" && main(ARGS)
else
    !isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)
end

end # module
