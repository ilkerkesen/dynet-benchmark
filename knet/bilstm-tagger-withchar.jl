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
    w, srnns = initweights(atype, o[:HIDDEN_SIZE], length(w2i), ntags, nchars,
                           o[:WEMBED_SIZE], o[:CEMBED_SIZE], o[:MLP_SIZE])
    s = initstate(atype, o[:HIDDEN_SIZE], o[:WEMBED_SIZE])
    opt = optimizers(w, Adam)

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
                for sent in tst
                    seq, is_word = make_input(sent, w2i, c2i)
                    nwords = length(sent)
                    ypred,_ = predict(w, seq, is_word, srnns)
                    ypred = map(x->i2t[x], mapslices(indmax,Array(ypred),1))
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

                all_time > o[:TIMEOUT] && return
            end

            # train with instance
            seq, is_word = make_input(trn[k],w2i,c2i)
            out = make_output(trn[k],t2i)
            batch_loss = train!(w,seq,is_word,out,srnns,opt)
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
            push!(seq, convert(Array{Int32}, map(c->c2i[c], chars)))
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
    state[1] = zeros(hidden, batchsize)
    state[2] = zeros(hidden, batchsize)
    state[3] = zeros(div(wembed,2), batchsize)
    state[4] = zeros(div(wembed,2), batchsize)
    return map(s->convert(atype,s), state)
end

# init LSTM parameters
function initlstm(input, hidden, winit=0.01)
    w = winit*randn(4*hidden, input+hidden)
    b = zeros(4*hidden, 1)
    b[1:hidden] = 1
    return (w, b)
end

# initialize all weights of the language model
function initweights(
    atype, hidden, words, tags, chars, wembed, cembed, mlp, winit=0.01)
    w = Array{Any}(8)

    # init rnns
    srnn1, wrnn1 = rnninit(wembed, hidden; bidirectional=true)
    w[1] = wrnn1
    srnn2, wrnn2 = rnninit(cembed, div(wembed,2); bidirectional=true)
    w[2] = wrnn2

    # weight/bias params for MLP network
    w[3] = convert(atype, winit*randn(mlp, 2*hidden))
    w[4] = convert(atype, zeros(mlp, 1))
    w[5] = convert(atype, winit*randn(tags, mlp))
    w[6] = convert(atype, winit*randn(tags, 1))

    # word/char embeddings
    w[7] = convert(atype, winit*randn(wembed, words))
    w[8] = convert(atype, winit*randn(cembed, chars))
    return w, [srnn1, srnn2]
end

# loss function
function loss(w, seq, is_word, ygold, srnns)
    py, _ = predict(w,seq,is_word,srnns)
    return nll(py,ygold)
end

lossgradient = gradloss(loss)

function predict(w,seq,is_word,srnns)
    x = encoder(w,seq,is_word,srnns[2])
    r = srnns[1]; wr = w[1]
    wmlp, bmlp = w[3], w[4]
    wy, by = w[5], w[6]
    y, hy, cy = rnnforw(r,wr,x)
    y2 = reshape(y,size(y,1),size(y,2)*size(y,3))
    y3 = wmlp * y2 .+ bmlp
    return wy*y3.+by, hy, cy
end

# encoder - it generates embeddings
function encoder(w,seq,is_word,srnn)
    # embedding
    embed = Array{Any}(length(seq))
    inds = convert(Array{Int32}, seq[is_word])
    wembed = w[end-1][:,inds]
    wi = 1
    wr = w[2]; r = srnn

    for k = 1:length(seq)
        if is_word[k] # common word embed
            embed[k] = wembed[:,wi:wi]
            wi += 1
        else # rare word embed
            inds = seq[k]
            inds = reshape(inds, 1, length(inds))
            cembed = w[end][:,inds]
            y, hy, cy = rnnforw(r,wr,cembed; hy=true, cy=true)
            embed[k] = reshape(hy, length(hy), 1)
        end
    end
    embed = hcat(embed...)
    embed = reshape(embed, size(embed,1), 1, size(embed,2))
end

function train!(w,seq,is_word,ygold,srnns,opt)
    gloss, lossval = lossgradient(w, seq, is_word, ygold, srnns)
    for k = 1:length(w)
        gloss[k] != nothing && update!(w[k], gloss[k], opt[k])
    end
    return lossval*length(seq)
end

if VERSION >= v"0.5.0-dev+7720"
    PROGRAM_FILE=="knet/bilstm-tagger-withchar.jl" && main(ARGS)
else
    !isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)
end

end # module
