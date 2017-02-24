using Knet
using AutoGrad
using ArgParse

const train_file = "data/trees/train.txt"
const dev_file = "data/trees/dev.txt"
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

    # count words and build vocabulary
    l2i, w2i, labels, words = build_vocabs(trn)
    nwords = length(words); nlabels = length(labels)
    # build model
    # w = initweights(atype, o[:HIDDEN_SIZE], length(w2i), length(t2i),
    #                 o[:MLP_SIZE], o[:EMBED_SIZE])
    # s = initstate(atype, o[:HIDDEN_SIZE], o[:batchsize])
    # opt = initopt(w)

    # train bilstm tagger
    println("nwords=$nwords, nlabels=$nlabels"); flush(STDOUT)
    println("startup time: ", Int(now()-t00)*0.001); flush(STDOUT)
    return;
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
    return within_bracket(tokens)[1]
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
    items = []
    function helper(subtree)
        if isleaf(subtree)
            push!(items, subtree)
        else
            for child in subtree.children
                helper(child)
            end
        end
    end
    helper(t)
    return items
end

function nonterms(t::Tree)
    nodes = []
    function helper(subtree)
        if !isleaf(subtree)
            push!(nodes, subtree)
            map(helper, subtree.children)
        end
    end
    helper(t)
    return nodes
end

function tokenize_sexpr(sexpr)
    tokker = r" +|[()]|[^ ()]+"
    filter(t -> t != " ", matchall(tokker, sexpr))
end

function within_bracket(tokens, state=1)
    (label, state) = next(tokens, state)
    children = []
    while !done(tokens, state)
        (token, state) = next(tokens, state)
        if token == "("
            (child, state) = within_bracket(tokens, state)
            push!(children, child)
        elseif token == ")"
            return Tree(label, children), state
        else
            push!(children, Tree(token))
        end
    end
end

function build_vocabs(trees)
    words = Set()
    labels = Set()
    for tree in trees
        push!(words, map(t->t.label, leaves(tree))...)
        push!(labels, map(t->t.label, nonterms(tree))...)
    end
    push!(words, UNK)
    w2i, i2w = build_vocab(words)
    l2i, i2l = build_vocab(labels)
    return l2i, w2i, i2l, i2w
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
function initstate(atype, hidden, batchsize=1)
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
function loss(w, s0, tree)
    total = 0
    function helper(subtree)
        if length(subtree.children) == 1 && isleaf(subtree.children[1])
            t = subtree.children[1]
            ss = lstm(w,b,copy(s0)...,t.data)

            # softloss calculation
        end

        length(subtree.children) == 2 || error("...")
        (t1,t2) = subtree.children

    end
    if length(t.children) == 1
        isleaf(t.children[1]) || error(t.children[1])
        return
    end
end

# lstm without forget gate
function lstm(weight, bias, hidden, cell, input)
    gates   = hcat(input,hidden) * weight .+ bias
    hsize   = size(hidden,2)
    ingate  = sigm(gates[:,1:hsize])
    outgate = sigm(gates[:,1+hsize:2hsize])
    change  = sigm(gates[:,1+2hsize:3hsize])
    cell    = ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden,cell)
end

# lstm with two children forget gates, wf/bf -> forget gate parameters
function lstm(weight, bias, hiddens, cells, input, wf, bf)
    hidden  = sum(hiddens)
    hsize   = size(hidden,2)
    gates   = hcat(input,hidden) * weight .+ bias
    ingate  = sigm(gates[:,1:hsize])
    outgate = sigm(gates[:,1+hsize:2hsize])
    change  = sigm(gates[:,1+2hsize:3hsize])
    forgets = map(h->sigm(hcat(input,h)*wf .+ bf), hiddens)
    cell    = ingate .* change + sum(x->x[1].*x[2], zip(forgets,cells))
    hidden  = outgate .* cell
    return (hidden,cell)
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
