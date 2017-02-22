using Knet
using AutoGrad
using ArgParse

const train_file = "data/tags/train.txt"
const dev_file = "data/tags/dev.txt"
const UNK = "_UNK_"
t00 = now()

function main(args)
    s = ArgParseSettings()
    s.description = "Bidirectional LSTM Tagger in Knet"

    @add_arg_table s begin
        ("--gpu"; action=:store_true; help="use GPU or not")
        ("EMBED_SIZE"; arg_type=Int; help="embedding size")
        ("HIDDEN_SIZE"; arg_type=Int; help="hidden size")
        ("SPARSE"; arg_type=Int; help="sparse update 0/1")
        ("TIMEOUT"; arg_type=Int; help="max timeout")
        ("--train"; default=train_file; help="train file")
        ("--dev"; default=dev_file; help="dev file")
        ("--seed"; arg_type=Int; default=-1; help="random seed")
        ("--epochs"; arg_type=Int; default=100; help="epochs")
        ("--minoccur"; arg_type=Int; default=0)
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:seed] > 0 && srand(o[:seed])
    atype = o[:gpu] ? KnetArray{Float32} : Float32

    # read data
    trn = read_file(o[:train])
    tst = read_file(o[:dev])

    # get words and tags from train set
    words, tags = [], []
    for sample in trn
        for (word,tag) in sample
            push!(words, word)
            push!(tags, tag)
        end
    end

    # count words and build vocabulary
    wordcounts = count_words(words)
    nwords = length(wordcounts)+1
    wordcounts = filter((w,c)-> c >= o[:minoccur], wordcounts)
    words = collect(keys(wordcounts))
    !in(UNK, words) && push!(words, UNK)
    w2i, i2w = build_vocabulary(words)
    t2i, i2t = build_vocabulary(tags)
    ntags = length(t2i)
    !haskey(w2i, UNK) && error("...")

    # build model
    w = initweights(atype, o[:HIDDEN_SIZE], length(w2i), length(t2i),
                    o[:MLP_SIZE], o[:EMBED_SIZE])
    s = initstate(atype, o[:HIDDEN_SIZE], o[:batchsize])
    opt = initopt(w)

    # train bilstm tagger
    println("nwords=$nwords, ntags=$ntags"); flush(STDOUT)
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
                for sent in tst
                    seq = make_input(sent, w2i)
                    nwords = length(sent)
                    ypred = predict(w, copy(s), seq)
                    ypred = map(x->i2t[x], predict(w,copy(s),seq))
                    ygold = map(x -> x[2], sent)
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

                all_time > o[:TIMEOUT] && exit()
            end

            # train on minibatch
            seq = make_input(trn[k], w2i)
            out = make_output(trn[k], t2i)

            batch_loss = train!(w,s,seq,out,opt)
            this_loss += batch_loss
            this_tagged += length(trn[k])
        end
        @printf("epoch %d finished\n", epoch-1); flush(STDOUT)
    end
end

# read file
function read_file(file)
    data = open(file, "r") do f
        map(parse_line, readlines(f))
    end
end

# parse line
function parse_line(line)
    ln = replace(line, "\n", "")
    tokens = tokenize_sexpr(ln)
    shift!(tokens)
    return within_bracket(tokens)
end

type Tree
    label
    children
    data
end

function Tree(x)
    return Tree(x,nothing,nothing)
end

function Tree(x,y)
    return Tree(x,y,nothing)
end

function isleaf(t::Tree)
    return t.children == nothing
end

function pretty(t::Tree)
    t.children == nothing && return t.label
    join([t.label; map(pretty, t.children)], " ")
end

function leaves(t::Tree)
    t.children == nothing && return t
    mapreduce(leaves_iter, vcat, t.children)
end

function nonterms(t::Tree)
    if !isleaf(t)
        return [t; mapreduce(nonterms, vcat, filter(c->!isleaf(c), t.children))]
    end
end

function tokenize_sexpr(sexpr)
    tokker = r" +|[()]|[^ ()]+"
    filter(t -> t != " ", matchall(tokker, sexpr))
end

function within_bracket(tokens)
    label = shift!(tokens)
    children = []
    while !isempty(tokens)
        token = shift!(tokens)
        if token == "("
            push!(children, _within_bracket(tokens))
        elseif token == ")"
            return Tree(label, children)
        else
            push!(children, Tree(label, nothing))
        end
    end
end

function count_words(words)
    wordcounts = Dict()
    for word in words
        wordcounts[word] = get(wordcounts, word, 0) + 1
    end
    return wordcounts
end

function build_vocabs(trees)
    words = Set()
    labels = Set()
    for tree in trees
        push!(words, map(t->t.label, leaves(tree)))
        push!(labels, map(t->t.label, nonterms(tree)))
    end
    w2i, _ = build_vocab(words)
    l2i, _ = build_bocab(labels)
    return l2i, w2i, labels, words
end

function build_vocab(xs)
    x2i = Dict(); i2x = Dict()
    for (i,x) in enumerate(xs)
        x2i[x] = i
        i2x[i] = x
    end
    return x2i, i2x
end

# initialize hidden and cell arrays
function initstate(atype, hidden, batchsize)
    state = Array(Any, 2)
    state[1] = zeros(batchsize, hidden)
    state[2] = zeros(batchsize, hidden)
    return map(s->convert(atype,s), state)
end

# initialize all weights of the language model
# w[1:2] => weight/bias params for forward LSTM network
# w[3:4] => weight/bias params for backward LSTM network
# w[5:8] => weight/bias params for MLP network
# w[9]   => word embeddings
function initweights(atype, hidden, words, tags, embed, mlp, winit=0.01)
    w = Array(Any, 9)
    input = embed
    w[1] = winit*randn(input+hidden, 4*hidden)
    w[2] = zeros(1, 4*hidden)
    w[2][1:hidden] = 1
    w[3] = winit*randn(input+hidden, 4*hidden)
    w[4] = zeros(1, 4*hidden)
    w[4][1:hidden] = 1
    w[5] = winit*randn(2*hidden, mlp)
    w[6] = zeros(1, mlp)
    w[7] = winit*randn(mlp, tags)
    w[8] = winit*randn(1, tags)
    w[9] = winit*randn(words, embed)
    return map(i->convert(atype, i), w)
end

# init optimization parameters (only ADAM with defaults)
initopt(w) = map(Adam, w)

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


# treenn loss function
function loss(w, s, tree)
end

# loss function
function loss(w, s, seq, out, values=[])
    # encoder
    sfs, sbs = encoder(w,s,seq)

    # prediction
    atype = typeof(AutoGrad.getval(w[1]))
    total = 0
    rng = 1:length(seq)
    for t in rng
        x = hcat(sfs[t], sbs[t]) * w[5] .+ w[6]
        ypred = x * w[7] .+ w[8]
        ynorm = logp(ypred,2)
        ygold = convert(atype, out[t])
        total += sum(ygold .* ynorm)
    end

    push!(values, AutoGrad.getval(-total))
    return -total
end

# bilstm encoder
function encoder(w,s,seq)
    atype = typeof(AutoGrad.getval(w[1]))

    # states and state histories
    sf, sb = copy(s), copy(s)
    sfs = Array(Any, length(seq))
    sbs = Array(Any, length(seq))

    # embedding
    embed = Array(Any, length(seq))
    for k = 1:length(seq)
        embed[k] = convert(atype, seq[k]) * w[9]
    end

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

# tag given input sentence
function predict(w,s,seq)
    # encoder
    sfs, sbs = encoder(w,s,seq)

    # prediction
    tags = []
    atype = typeof(AutoGrad.getval(w[1]))
    rng = 1:length(seq)
    for t in rng
        x = hcat(sfs[t], sbs[t]) * w[5] .+ w[6]
        ypred = x * w[7] .+ w[8]
        ypred = convert(Array{Float32}, ypred)[:]
        push!(tags, indmax(ypred))
    end

    tags
end

lossgradient = grad(loss)

function train!(w,s,seq,tags,opt)
    values = []
    gloss = lossgradient(w, copy(s), seq, tags, values)
    for k = 1:length(w)
        update!(w[k], gloss[k], opt[k])
    end
    isa(s,Vector{Any}) || error("State should not be Boxed.")
    for k = 1:length(s)
        s[k] = AutoGrad.getval(s[k])
    end
    values[1]
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
