#include <math.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
class LayerNormalizationImpl : public torch::nn::Module {
   public:
    LayerNormalizationImpl(int features, float eps = 1e-6)
        : alpha(torch::ones(features)), bias(torch::zeros(features)), eps(eps) {
        register_parameter("alpha", alpha);
        register_parameter("bias", bias);
    }

    torch::Tensor forward(torch::Tensor x) {
        auto mean = x.mean(-1, true);
        auto std = x.std(-1, true);
        return alpha * (x - mean) / (std + eps) + bias;
    }

    torch::Tensor alpha, bias;
    float eps;
};
TORCH_MODULE(LayerNormalizationImpl);

class FeedForwardBlockImpl : public torch::nn::Module {
   public:
    FeedForwardBlockImpl(int d_model, int d_ff, float dropout)
        : linear1(torch::nn::Linear(d_model, d_ff)),
          dropout(torch::nn::Dropout(dropout)),
          linear2(torch::nn::Linear(d_ff, d_model)) {
        register_module("linear1", linear1);
        register_module("dropout", dropout);
        register_module("linear2", linear2);
    }
    torch::Tensor forward(torch::Tensor x) {
        return linear2->forward(dropout->forward(torch::relu(linear1->forward(x))));
    }

    torch::nn::Linear linear1, linear2;
    torch::nn::Dropout dropout;
};


TORCH_MODULE(FeedForwardBlockImpl);

class InputEmbeddingsImpl : public torch::nn::Module {
   public:
    InputEmbeddingsImpl(int d_model, int vocab_size)
        : embedding(torch::nn::Embedding(vocab_size, d_model)),
          d_model(d_model) {
        register_module("embedding", embedding);
    }
    torch::Tensor forward(torch::Tensor x) {
        return embedding->forward(x) * std::sqrt(d_model);
    }

    torch::nn::Embedding embedding;
    int d_model;
};

TORCH_MODULE(InputEmbeddingsImpl);

class PositionalEncodingImpl : public torch::nn::Module {
   public:
    PositionalEncodingImpl(int d_model, int seq_len, float dropout)
        : d_model(d_model),
          seq_len(seq_len),
          dropout(torch::nn::Dropout(dropout)) {
        auto pe = torch::zeros({seq_len, d_model});
        auto position = torch::arange(seq_len, torch::kFloat).unsqueeze(1);
        auto div_term = torch::exp(torch::arange(0, d_model, 2, torch::kFloat) *
                                   (-std::log(10000.0) / d_model));
        pe.index_put_({torch::Slice(), torch::Slice(0, torch::None, 2)},
                      torch::sin(position * div_term));
        pe.index_put_({torch::Slice(), torch::Slice(1, torch::None, 2)},
                      torch::cos(position * div_term));
        pe = pe.unsqueeze(0);
        register_buffer("pe", pe);  // Register the postional as a buffer
    }

    torch::Tensor forward(torch::Tensor x) {
        x = x + pe.index({torch::Slice(None), torch::Slice()});
        return dropout->forward(x);
    }

    torch::Tensor pe;
    int d_model, seq_len;
    torch::nn::Dropout dropout;
};

TORCH_MODULE(PositionalEncodingImpl);

class ResidualConnectionImpl : public torch::nn::Module {
   public:
    ResidualConnectionImpl(int features, float dropout)
        : dropout(torch::nn::Dropout(dropout)),
          norm(LayerNormalizationImpl(features)) {
        register_module("dropout", dropout);
        register_module("norm", norm);
    }

    torch::Tensor forward(torch::Tensor x,
                          torch::Tensor (*sublayer)(torch::Tensor)) {
        return x + dropout->forward(sublayer(norm->forward(x)));
    }

    torch::nn::Dropout dropout;
    LayerNormalizationImpl norm;
};

TORCH_MODULE(ResidualConnectionImpl);
class MultiHeadAttentionBlockImpl : public torch::nn::Module {
   public:
    MultiHeadAttentionBlockImpl(int d_model, int h, float dropout)
        : d_model(d_model),
          h(h),
          d_k(d_model / h),
          d_h(d_model / h),
          d_v(d_model / h),
          w_q(torch::nn::Linear(d_model, d_model).without_bias()),
          w_k(torch::nn::Linear(d_model, d_model).without_bias()),
          w_v(torch::nn::Linear(d_model, d_model).without_bias()),
          w_o(torch::nn::Linear(d_model, d_model).without_bias()),
          dropout(torch::nn::Dropout(dropout)) {
        register_module("w_q", w_q);
        register_module("w_k", w_k);
        register_module("w_v", w_v);
        register_module("w_o", w_o);
        register_module("dropout", dropout);
    }

    torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                          torch::Tensor mask) {
        q = w_q->forward(q).view({q.size(0), q.size(1), h, d_h}).transpose(1, 2);
        k = w_k->forward(k).view({k.size(0), k.size(1), h, d_k}).transpose(1, 2);
        v = w_v->forward(v).view({v.size(0), v.size(1), h, d_v}).transpose(1, 2);
        auto attn_output = attention(q, k, v, mask, dropout);
        auto x = std::get<0>(attn_output)
                     .transpose(1, 2)
                     .contiguous()
                     .view({-1, q.size(2) * d_k});
        return w_o->forward(x);
    }

    std::tuple<torch::Tensor, torch::Tensor> attention(
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor mask, torch::nn::Dropout dropout) {
        auto d_k = query.size(-1);
        auto scores = torch::matmul(query, key.transpose(-2, -1)) / std::sqrt(d_k);
        if (!mask.is_empty()) {
            scores.masked_fill_(mask, -1e9);
        }
        auto attn_weights = torch::softmax(scores, -1);
        if (dropout->p() > 0.0) {
            attn_weights = dropout->forward(attn_weights);
        }
        auto output = torch::matmul(attn_weights, value);
        return std::make_tuple(output, attn_weights);
    }

    int d_model, h, d_k, d_h, d_v;
    torch::nn::Linear w_q, w_k, w_v, w_o;
    torch::nn::Dropout dropout;
};


TORCH_MODULE(MultiHeadAttentionBlockImpl);

class EncoderBlockImpl : public torch::nn::Module {
   public:
    EncoderBlockImpl(int features, MultiHeadAttentionBlockImpl self_attention_block,
                     FeedForwardBlockImpl feed_forward_block, float dropout)
        : self_attention_block(std::move(self_attention_block)),
          feed_forward_block(std::move(feed_forward_block)),
          residual_connections(
              register_module("residual_connections",
                              torch::nn::ModuleList<ResidualConnectionImpl>(2))) {}

    torch::Tensor forward(torch::Tensor x, torch::Tensor src_mask) {
        x = residual_connections[0]->forward(x, [&](torch::Tensor x) {
            return self_attention_block->forward(x, x, x, src_mask);
        });
        return residual_connections[1]->forward(x, feed_forward_block);
    }

    torch::nn::AnyModule self_attention_block, feed_forward_block;
    torch::nn::ModuleList<ResidualConnectionImpl> residual_connections;
};

TORCH_MODULE(EncoderBlockImpl);

class EncoderImpl : public torch::nn::Module {
   public:
    EncoderImpl(int features, torch::ModuleList<EncoderBlockImpl> layers)
        : layers(std::move(layers)), norm(LayerNormalizationImpl(features)) {
        register_module("norm", norm);
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor mask) {
        for (auto& layer : layers) {
            x = layer->forward(x, mask);
        }
        return norm->forward(x);
    }

    torch::nn::ModuleList<EncoderBlockImpl> layers;
    LayerNormalizationImpl norm;
};


TORCH_MODULE(EncoderImpl);

class DecoderBlockImpl : public torch::nn::Module {
    public:
    DecoderBlockImpl(int features, MultiHeadAttentionBlockImpl self_attention_block, MultiHeadAttentionBlockImpl cross_attention_block, FeedForwardBlockImpl feed_forward_block, float dropout)
    : self_attention_block(std::move(self_attention_block)), cross_attention_block(std::move(cross_attention_block)), feed_forward_block(std::move(feed_forward_block)),
    residual_connections(register_module("residual_connections", torch::nn::ModuleList<ResidualConnectionImpl>(3))) {}

    torch::Tensor forward(torch::Tensor x, torch::Tensor encoder_output, torch::Tensor encoder_output, torch::Tensor src_mask, torch::Tensor tgt_mask) {
        x = residual_connections[0]->forward(x, [&](torch::Tensor x) { return self_attention_block.forward(x, x, x, tgt_mask); });
        x = residual_connections[1]->forward(x, [&](torch::Tensor x) { return cross_attention_block.forward(x, encoder_output, encoder_output, src_mask); });
        return residual_connections[2]->forward(x, feed_forward_block);
    }

    torch::nn::AnyModule self_attention_block, cross_attention_block, feed_forward_block;
    torch::nn::ModuleList<ResidualConnectionImpl> residual_connections;
};

TORCH_MODULE(DecoderBlockImpl);

class DecoderImpl : public torch::nn::Module {
    public:
    DecoderImpl(int features, int layers, torch::nn::ModuleList<DecoderBlockImpl> layers)
    : layers(std::move(layers)), norm(LayerNormalizationImpl(features)) {
        register_module("norm", norm);
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor encoder_output, torch::Tensor src_mask, torch::Tensor tgt_mask) {
        for(auto& layer : layers) {
            x = layer(x, encoder_output, src_mask, tgt_mask);
        }
        return norm(x);
    }

    torch::nn::ModuleList<DecoderBlockImpl> layers;
    LayerNormalizationImpl norm;
};

TORCH_MODULE(DecoderImpl);



class ProjectionLayerImpl : public torch::nn::Module {
public:
    ProjectionLayerImpl(int d_model, int vocab_size)
        : proj(torch::nn::Linear(d_model, vocab_size)) {
        register_module("proj", proj);
    }

    torch::Tensor forward(torch::Tensor x) {
        return proj->forward(x);
    }

    torch::nn::Linear proj;
};


TORCH_MODULE(ProjectionLayer);

class TransformerImpl : public torch::nn::Module {
    public:
    TransformerImpl(EncoderImpl encoder, DecoderImpl decoder,
    InputEmbeddingsImpl src_embed, InputEmbeddingsImpl tgt_embed,
    PositionalEncodingImpl src_pos, PositionalEncodingImpl tgt_pos,
    ProjectionLayerImpl projection_layer)
    : encoder(std::move(encoder)), decoder(std::move(decoder)),
    src_embed(std::move(src_embed)), tgt_embed(std::move(tgt_embed)),
    src_pos(std::move(src_pos)), tgt_pos(std::move(tgt_pos)),
    projection_layer(std::move(projection_layer)) {}

    torch::Tensor encode(torch::Tensor src, torch::Tensor src_mask) {
        src = src_embed.forward(src);
        src = src_pos.forward(src);
        return encoder.forward(src, src_mask);
    }

    torch::Tensor decode(torch::Tensor encoder_output, torch::Tensor src_mask, torch::Tensor tgt, torch::Tensor tgt_mask) {
        tgt = tgt_embed.forward(tgt);
        tgt = tgt_pos.forward(tgt);
        return decoder.forward(tgt, encoder_output, src_mask, tgt_mask);
    }

    torch::Tensor project(torch::Tensor x) {
        return projection_layer.forward(x);
    }

    torch::nn::AnyModule encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer;
};

TORCH_MODULE(TransformerImpl);

It seems there are still some issues in the code. Let's address them:

    The error ‘torch::nn::ModuleList’ is not a template suggests that ModuleList is being used incorrectly. It should be used without template parameters.

    The error ‘class torch::nn::ModuleList’ has no member named ‘push_back’ indicates that ModuleList does not have a push_back method. Instead, you should use the append method.

Here's the corrected version of the build_transformer function:

cpp

TransformerImpl build_transformer(int src_vocab_size, int tgt_vocab_size, int src_seq_len, int tgt_seq_len, int d_model=512, int N=6, int h=8, float dropout=0.1, int d_ff=2048) {
    auto src_embed = InputEmbeddingsImpl(d_model, src_vocab_size);
    auto tgt_embed = InputEmbeddingsImpl(d_model, tgt_vocab_size);
    auto src_pos = PositionalEncodingImpl(d_model, src_seq_len, dropout);
    auto tgt_pos = PositionalEncodingImpl(d_model, tgt_seq_len, dropout);

    torch::nn::ModuleList encoder_blocks;
    for (int i = 0; i < N; ++i) {
        auto encoder_self_attention_block = MultiHeadAttentionBlockImpl(d_model, h, dropout);
        auto feed_forward_block = FeedForwardBlockImpl(d_model, d_ff, dropout);
        encoder_blocks.append(EncoderBlockImpl(d_model, encoder_self_attention_block, feed_forward_block, dropout));
    }

    torch::nn::ModuleList decoder_blocks;
    for (int i = 0; i < N; ++i) {
        auto decoder_self_attention_block = MultiHeadAttentionBlockImpl(d_model, h, dropout);
        auto decoder_cross_attention_block = MultiHeadAttentionBlockImpl(d_model, h, dropout);
        auto feed_forward_block = FeedForwardBlockImpl(d_model, d_ff, dropout);
        decoder_blocks.append(DecoderBlockImpl(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout));
    }

    auto encoder = EncoderImpl(d_model, encoder_blocks);
    auto decoder = DecoderImpl(d_model, N, decoder_blocks);
    auto projection_layer = ProjectionLayerImpl(d_model, tgt_vocab_size);

    TransformerImpl transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer);

    for (auto& p : transformer.parameters()) {
        if (p.dim() > 1) {
            torch::nn::init::xavier_uniform_(p);
        }
    }
    
    return transformer;
}

int main() {
    auto transformer = build_transformer(10000, 10000, 50, 50);
    torch::Tensor src = torch::randn({64, 50}).to(torch::kLong);
    torch::Tensor tgt = torch::randn({64, 50}).to(torch::kLong);
    torch::Tensor src_mask = torch::ones({1, 1, 50, 50}).to(torch::kByte);
    torch::Tensor tgt_mask = torch::ones({1, 1, 50, 50}).to(torch::kByte);
    // Encode the source tensor
    auto encoder_output = transformer.encode(src, src_mask);
    std::cout << "Encoder output:\n" << encoder_output << std::endl;

    // Decode the encoder output
    auto output = transformer.decode(encoder_output, src_mask, tgt, tgt_mask);
    std::cout << "Decoder output:\n" << output << std::endl;

    // Project the output
    auto proj_output = transformer.project(output);
    std::cout << "Projected output:\n" << proj_output << std::endl;
    return 0;
}

