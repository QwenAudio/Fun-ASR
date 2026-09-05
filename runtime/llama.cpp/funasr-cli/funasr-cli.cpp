// funasr-cli: end-to-end Fun-ASR-Nano in C++ on the llama.cpp / ggml stack.
//
//   wav(16k mono) -> kaldi fbank -> SAN-M encoder + adaptor (ggml) ->
//   low-frame-rate truncation -> [prefix tokens | audio embeds | suffix tokens]
//   -> Qwen3 LLM (llama.cpp) -> transcription.
//
// This is the whisper.cpp-style single-binary path: no Python at runtime.
//
//   funasr-cli --enc funasr-encoder.gguf -m qwen3-0.6b.gguf -a audio.wav
//
// Streaming (--stream): 16 kHz s16le mono PCM on stdin, models load once, emits
//   LOCKED <text>   DynamicStreamingVAD-confirmed segment (committed, like python)
//   PARTIAL <text>  unstable transcript of the in-progress segment (or empty when it ends)
//   DONE            after stdin EOF + trailing-segment flush
// Thresholds/protocol mirror serve_realtime_ws.py (RealtimeASRSession): 480ms first
// decode, 960ms cadence, 15s partial window cap, 200/512 token budgets for partial/locked.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// any audio (wav/mp3/flac, any rate/channels) -> 16 kHz mono f32, via miniaudio
#define FUNASR_AUDIO_IMPLEMENTATION
#include "funasr_audio.h"
// built-in FSMN-VAD front end (single-binary --vad segmentation)
#include "funasr_vad.h"
#include <utility>

// ======================= kaldi fbank + LFR =======================
static const int FS=16000, WINLEN=400, SHIFT=160, NFFT=512, NMEL=80, LFR_M=7, LFR_N=6;
static const float PREEMPH=0.97f, LOWF=20.0f, HIGHF=8000.0f;
static inline float mel(float f){ return 1127.0f*logf(1.0f+f/700.0f); }
static void fft(std::vector<float>&re,std::vector<float>&im,int n){
    for(int i=1,j=0;i<n;i++){int b=n>>1;for(;j&b;b>>=1)j^=b;j^=b;if(i<j){std::swap(re[i],re[j]);std::swap(im[i],im[j]);}}
    for(int len=2;len<=n;len<<=1){double a=-2.0*M_PI/len;float wr=cosf(a),wi=sinf(a);
        for(int i=0;i<n;i+=len){float cr=1,ci=0;for(int k=0;k<len/2;k++){
            float ur=re[i+k],ui=im[i+k];float vr=re[i+k+len/2]*cr-im[i+k+len/2]*ci,vi=re[i+k+len/2]*ci+im[i+k+len/2]*cr;
            re[i+k]=ur+vr;im[i+k]=ui+vi;re[i+k+len/2]=ur-vr;im[i+k+len/2]=ui-vi;
            float n2=cr*wr-ci*wi;ci=cr*wi+ci*wr;cr=n2;}}}
}
// returns [T x 560] row-major, sets T
static std::vector<float> compute_fbank(std::vector<float> wav, int & T_out) {
    for (auto & v : wav) v *= 32768.0f;
    std::vector<float> win(WINLEN);
    for (int i=0;i<WINLEN;i++) win[i]=0.54f-0.46f*cosf(2.0f*M_PI*i/(WINLEN-1));
    const int NBIN=NFFT/2+1; float bw=(float)FS/NFFT, ml=mel(LOWF), mh=mel(HIGHF), dm=(mh-ml)/(NMEL+1);
    std::vector<std::vector<float>> fb(NMEL, std::vector<float>(NBIN,0.0f));
    for(int m=0;m<NMEL;m++){float L=ml+m*dm,C=ml+(m+1)*dm,R=ml+(m+2)*dm;
        for(int k=0;k<NBIN;k++){float mf=mel(bw*k); if(mf>L&&mf<R) fb[m][k]=mf<=C?(mf-L)/(C-L):(R-mf)/(R-C);}}
    int N=wav.size(); int T=(N-WINLEN)/SHIFT+1;
    std::vector<std::vector<float>> feat(T, std::vector<float>(NMEL));
    std::vector<float> re(NFFT),im(NFFT),fr(WINLEN);
    const float fl=1.1920929e-07f;
    for(int t=0;t<T;t++){const float*s=wav.data()+t*SHIFT;
        double mn=0;for(int i=0;i<WINLEN;i++)mn+=s[i];mn/=WINLEN;
        for(int i=0;i<WINLEN;i++)fr[i]=s[i]-(float)mn;
        for(int i=WINLEN-1;i>0;i--)fr[i]-=PREEMPH*fr[i-1];fr[0]-=PREEMPH*fr[0];
        for(int i=0;i<NFFT;i++){re[i]=i<WINLEN?fr[i]*win[i]:0.0f;im[i]=0.0f;}
        fft(re,im,NFFT);
        for(int m=0;m<NMEL;m++){float e=0;for(int k=0;k<NBIN;k++)if(fb[m][k]>0)e+=fb[m][k]*(re[k]*re[k]+im[k]*im[k]);
            feat[t][m]=logf(e>fl?e:fl);}}
    // LFR
    const int pad=(LFR_M-1)/2; int T_lfr=(T+LFR_N-1)/LFR_N;
    std::vector<std::vector<float>> pd; pd.reserve(T+pad+LFR_M);
    for(int i=0;i<pad;i++)pd.push_back(feat[0]);
    for(int t=0;t<T;t++)pd.push_back(feat[t]);
    while((int)pd.size()<(T_lfr-1)*LFR_N+LFR_M)pd.push_back(feat[T-1]);
    int D=LFR_M*NMEL; std::vector<float> out((size_t)T_lfr*D);
    for(int i=0;i<T_lfr;i++)for(int j=0;j<LFR_M;j++)
        memcpy(&out[(size_t)i*D+j*NMEL],pd[i*LFR_N+j].data(),NMEL*sizeof(float));
    T_out=T_lfr; return out;
}

// ======================= ggml SAN-M encoder + adaptor =======================
struct cfg { int d_model=512,n_head=4,num_blocks=50,tp_blocks=20,kernel=11,adp_llm=1024,adp_layers=2,adp_head=8; };
struct enc_model { cfg c; ggml_context*ctx_w=nullptr; std::map<std::string,ggml_tensor*> t;
    ggml_tensor* g(const std::string&n){auto it=t.find(n);if(it==t.end()){fprintf(stderr,"missing %s\n",n.c_str());exit(1);}return it->second;} };
static const float LN_EPS=1e-5f;
static bool load_enc(const char*p, enc_model&m){
    gguf_init_params gp={false,&m.ctx_w}; gguf_context*g=gguf_init_from_file(p,gp); if(!g)return false;
    auto rd=[&](const char*k,int d){int i=gguf_find_key(g,k);return i<0?d:(int)gguf_get_val_u32(g,i);};
    m.c.d_model=rd("funasr.enc.output_size",512); m.c.n_head=rd("funasr.enc.attention_heads",4);
    m.c.num_blocks=rd("funasr.enc.num_blocks",50); m.c.tp_blocks=rd("funasr.enc.tp_blocks",20);
    m.c.kernel=rd("funasr.enc.kernel_size",11); m.c.adp_llm=rd("funasr.adp.llm_dim",1024);
    m.c.adp_layers=rd("funasr.adp.n_layer",2); m.c.adp_head=rd("funasr.adp.attention_heads",8);
    int n=gguf_get_n_tensors(g); for(int i=0;i<n;i++){const char*nm=gguf_get_tensor_name(g,i);m.t[nm]=ggml_get_tensor(m.ctx_w,nm);}
    gguf_free(g); return true;
}
static ggml_tensor* lin(ggml_context*c,ggml_tensor*w,ggml_tensor*b,ggml_tensor*x){auto y=ggml_mul_mat(c,w,x);return b?ggml_add(c,y,b):y;}
static ggml_tensor* lnorm(ggml_context*c,ggml_tensor*x,ggml_tensor*g,ggml_tensor*b){return ggml_add(c,ggml_mul(c,ggml_norm(c,x,LN_EPS),g),b);}
static ggml_tensor* sanm_attn(ggml_context*c,enc_model&m,const std::string&p,ggml_tensor*x,int T){
    const int D=m.c.d_model,H=m.c.n_head,dk=D/H,K=m.c.kernel;
    ggml_tensor*qkv=lin(c,m.g(p+"linear_q_k_v.weight"),m.g(p+"linear_q_k_v.bias"),x); size_t nb1=qkv->nb[1];
    ggml_tensor*q=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,0));
    ggml_tensor*k=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,(size_t)D*sizeof(float)));
    ggml_tensor*v=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,(size_t)2*D*sizeof(float)));
    const int pad=(K-1)/2; ggml_tensor*fk=m.g(p+"fsmn_block.weight");
    ggml_tensor*vp=ggml_pad_ext(c,v,0,0,pad,pad,0,0,0,0); ggml_tensor*fsmn=v;
    for(int j=0;j<K;j++){auto sl=ggml_view_2d(c,vp,D,T,vp->nb[1],(size_t)j*vp->nb[1]);
        auto wj=ggml_view_1d(c,fk,D,(size_t)j*fk->nb[1]); fsmn=ggml_add(c,fsmn,ggml_mul(c,ggml_cont(c,sl),wj));}
    q=ggml_permute(c,ggml_reshape_3d(c,q,dk,H,T),0,2,1,3); k=ggml_permute(c,ggml_reshape_3d(c,k,dk,H,T),0,2,1,3);
    ggml_tensor*vh=ggml_cont(c,ggml_permute(c,ggml_reshape_3d(c,v,dk,H,T),1,2,0,3));
    ggml_tensor*kq=ggml_soft_max(c,ggml_scale(c,ggml_mul_mat(c,k,q),1.0f/sqrtf((float)dk)));
    ggml_tensor*o=ggml_cont_2d(c,ggml_permute(c,ggml_mul_mat(c,vh,kq),0,2,1,3),D,T);
    return ggml_add(c,lin(c,m.g(p+"linear_out.weight"),m.g(p+"linear_out.bias"),o),fsmn);
}
static ggml_tensor* sanm_layer(ggml_context*c,enc_model&m,const std::string&p,ggml_tensor*x,int T,bool res){
    auto r=x; auto h=lnorm(c,x,m.g(p+"norm1.weight"),m.g(p+"norm1.bias"));
    auto sa=sanm_attn(c,m,p+"self_attn.",h,T); x=res?ggml_add(c,r,sa):sa; r=x;
    h=lnorm(c,x,m.g(p+"norm2.weight"),m.g(p+"norm2.bias"));
    h=lin(c,m.g(p+"feed_forward.w_1.weight"),m.g(p+"feed_forward.w_1.bias"),h); h=ggml_relu(c,h);
    h=lin(c,m.g(p+"feed_forward.w_2.weight"),m.g(p+"feed_forward.w_2.bias"),h); return ggml_add(c,r,h);
}
static ggml_tensor* adp_layer(ggml_context*c,enc_model&m,const std::string&p,ggml_tensor*x,int T){
    const int D=m.c.adp_llm,H=m.c.adp_head,dk=D/H; auto r=x;
    auto h=lnorm(c,x,m.g(p+"norm1.weight"),m.g(p+"norm1.bias"));
    auto q=ggml_permute(c,ggml_reshape_3d(c,lin(c,m.g(p+"self_attn.linear_q.weight"),m.g(p+"self_attn.linear_q.bias"),h),dk,H,T),0,2,1,3);
    auto k=ggml_permute(c,ggml_reshape_3d(c,lin(c,m.g(p+"self_attn.linear_k.weight"),m.g(p+"self_attn.linear_k.bias"),h),dk,H,T),0,2,1,3);
    auto vh=ggml_cont(c,ggml_permute(c,ggml_reshape_3d(c,lin(c,m.g(p+"self_attn.linear_v.weight"),m.g(p+"self_attn.linear_v.bias"),h),dk,H,T),1,2,0,3));
    auto kq=ggml_soft_max(c,ggml_scale(c,ggml_mul_mat(c,k,q),1.0f/sqrtf((float)dk)));
    auto o=ggml_cont_2d(c,ggml_permute(c,ggml_mul_mat(c,vh,kq),0,2,1,3),D,T);
    x=ggml_add(c,r,lin(c,m.g(p+"self_attn.linear_out.weight"),m.g(p+"self_attn.linear_out.bias"),o)); r=x;
    h=lnorm(c,x,m.g(p+"norm2.weight"),m.g(p+"norm2.bias"));
    h=lin(c,m.g(p+"feed_forward.w_1.weight"),m.g(p+"feed_forward.w_1.bias"),h); h=ggml_relu(c,h);
    h=lin(c,m.g(p+"feed_forward.w_2.weight"),m.g(p+"feed_forward.w_2.bias"),h); return ggml_add(c,r,h);
}
static void add_posenc(std::vector<float>&x,int T,int depth){
    double inc=log(10000.0)/(depth/2.0-1.0);
    for(int t=0;t<T;t++){double pos=t+1;for(int i=0;i<depth/2;i++){double its=exp(i*-inc),st=pos*its;
        x[(size_t)t*depth+i]+=(float)sin(st);x[(size_t)t*depth+depth/2+i]+=(float)cos(st);}}
}
// fbank [T x F] -> adaptor out [T x adp_llm] row-major
static std::vector<float> run_encoder(enc_model&m,std::vector<float> fbank,int T,int F,int&Dout){
    float sc=sqrtf((float)m.c.d_model); for(auto&v:fbank)v*=sc; add_posenc(fbank,T,F);
    ggml_backend_t be=ggml_backend_cpu_init();
    ggml_init_params cp={(size_t)1024*1024*1024,nullptr,true}; ggml_context*c=ggml_init(cp);
    ggml_tensor*inp=ggml_new_tensor_2d(c,GGML_TYPE_F32,F,T); ggml_set_input(inp);
    ggml_tensor*x=sanm_layer(c,m,"audio_encoder.encoders0.0.",inp,T,false);
    for(int i=0;i<m.c.num_blocks-1;i++) x=sanm_layer(c,m,"audio_encoder.encoders."+std::to_string(i)+".",x,T,true);
    x=lnorm(c,x,m.g("audio_encoder.after_norm.weight"),m.g("audio_encoder.after_norm.bias"));
    for(int i=0;i<m.c.tp_blocks;i++) x=sanm_layer(c,m,"audio_encoder.tp_encoders."+std::to_string(i)+".",x,T,true);
    x=lnorm(c,x,m.g("audio_encoder.tp_norm.weight"),m.g("audio_encoder.tp_norm.bias"));
    x=lin(c,m.g("audio_adaptor.linear1.weight"),m.g("audio_adaptor.linear1.bias"),x); x=ggml_relu(c,x);
    x=lin(c,m.g("audio_adaptor.linear2.weight"),m.g("audio_adaptor.linear2.bias"),x);
    for(int i=0;i<m.c.adp_layers;i++) x=adp_layer(c,m,"audio_adaptor.blocks."+std::to_string(i)+".",x,T);
    ggml_set_output(x);
    ggml_cgraph*gf=ggml_new_graph_custom(c,32768,false); ggml_build_forward_expand(gf,x);
    ggml_gallocr_t ga=ggml_gallocr_new(ggml_backend_cpu_buffer_type()); ggml_gallocr_alloc_graph(ga,gf);
    ggml_backend_tensor_set(inp,fbank.data(),0,ggml_nbytes(inp));
    ggml_backend_cpu_set_n_threads(be,8); ggml_backend_graph_compute(be,gf);
    Dout=(int)x->ne[0]; std::vector<float> out((size_t)Dout*T); ggml_backend_tensor_get(x,out.data(),0,ggml_nbytes(x));
    ggml_gallocr_free(ga); ggml_free(c); ggml_backend_free(be); return out;
}

// ======================= LLM (llama.cpp) =======================
static int decode_batch(llama_context*ctx,int n,llama_token*tok,float*embd,int n_embd,int&n_past,bool last_logits){
    std::vector<llama_pos> pos(n); std::vector<int32_t> nsid(n,1);
    std::vector<llama_seq_id> s0(1,0); std::vector<llama_seq_id*> sid(n); std::vector<int8_t> lg(n,0);
    for(int i=0;i<n;i++){pos[i]=n_past+i;sid[i]=s0.data();}
    if(last_logits) lg[n-1]=1;
    llama_batch b={n,tok,embd,pos.data(),nsid.data(),sid.data(),lg.data()};
    int r=llama_decode(ctx,b); n_past+=n; return r;
}

// ---- encoder + LLM bundle, loaded once, reused per decode window ----
struct asr_decoder {
    enc_model em;
    llama_model* model=nullptr;
    const llama_vocab* vocab=nullptr;
    llama_context* ctx=nullptr;
    llama_sampler* smpl=nullptr;
    std::vector<llama_token> pre, suf;
};
static void free_decoder(asr_decoder&d){
    if(d.smpl) llama_sampler_free(d.smpl);
    if(d.ctx) llama_free(d.ctx);
    if(d.model) llama_model_free(d.model);
    if(d.em.ctx_w) ggml_free(d.em.ctx_w);
    d = asr_decoder{};
}
static bool load_decoder(const char*enc_path, const char*llm_path, float rep,
                         const std::string&lang, asr_decoder&d){
    if(!load_enc(enc_path,d.em)) return false;
    ggml_backend_load_all();
    llama_model_params mp=llama_model_default_params(); mp.n_gpu_layers=0;
    d.model=llama_model_load_from_file(llm_path,mp); if(!d.model) return false;
    d.vocab=llama_model_get_vocab(d.model);
    llama_context_params cp=llama_context_default_params();
    cp.n_ctx=2048; cp.n_batch=2048; cp.n_ubatch=2048;
    d.ctx=llama_init_from_model(d.model,cp);
    if(!d.ctx){fprintf(stderr,"failed to create llama context\n");return false;}
    auto sp=llama_sampler_chain_default_params(); d.smpl=llama_sampler_chain_init(sp);
    if(rep!=1.0f) llama_sampler_chain_add(d.smpl,llama_sampler_init_penalties(256,rep,0.0f,0.0f));
    llama_sampler_chain_add(d.smpl,llama_sampler_init_greedy());
    // prompt mirrors FunASRNano.get_prompt(): 语音转写：" / 语音转写成<lang>："
    std::string prefix="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n语音转写";
    if(!lang.empty()) prefix+="成"+lang;
    prefix+="：";
    const char*suffix="<|im_end|>\n<|im_start|>assistant\n";
    auto tokenize=[&](const std::string&s){int n=-llama_tokenize(d.vocab,s.c_str(),s.size(),nullptr,0,false,true);
        std::vector<llama_token> v(n); llama_tokenize(d.vocab,s.c_str(),s.size(),v.data(),n,false,true); return v;};
    d.pre=tokenize(prefix); d.suf=tokenize(suffix);
    return true;
}
// Decode one 16k-mono window (seconds*16000 samples, [-1,1]) into text.
static std::string asr_decode_window(asr_decoder&d, const float*pcm, size_t n, int npred){
    std::vector<float> seg(pcm, pcm+n);
    int T=0; auto fbank=compute_fbank(seg,T);
    int D=0; auto adp=run_encoder(d.em,fbank,T,560,D);
    int ol=1+(T-3+2)/2; ol=1+(ol-3+2)/2; int n_aud=(ol-1)/2+1;

    llama_memory_clear(llama_get_memory(d.ctx), true);  // fresh context per chunk
    int n_past=0;
    decode_batch(d.ctx,d.pre.size(),d.pre.data(),nullptr,0,n_past,false);
    decode_batch(d.ctx,n_aud,nullptr,adp.data(),D,n_past,false);
    decode_batch(d.ctx,d.suf.size(),d.suf.data(),nullptr,0,n_past,true);
    std::string out;
    llama_token tk=llama_sampler_sample(d.smpl,d.ctx,-1);
    for(int i=0;i<npred;i++){
        if(llama_vocab_is_eog(d.vocab,tk))break;
        char buf[256]; int k=llama_token_to_piece(d.vocab,tk,buf,sizeof(buf),0,true);
        if(k>0) out.append(buf,k);
        decode_batch(d.ctx,1,&tk,nullptr,0,n_past,true);
        tk=llama_sampler_sample(d.smpl,d.ctx,-1);
    }
    return out;
}

// ---- text cleanup, mirroring serve_realtime_ws.py ----
// _clean_asr_text(): strip <tags>, [brackets], a few artifact chars/tokens, collapse spaces.
static std::string clean_asr_text(const std::string& in){
    std::string s; s.reserve(in.size());
    for(size_t i=0;i<in.size();){
        char c=in[i];
        if(c=='<'){ size_t e=in.find('>',i); if(e!=std::string::npos){ i=e+1; continue; } }
        if(c=='['){ size_t e=in.find(']',i); if(e!=std::string::npos){ i=e+1; continue; } }
        if(c==']'||c=='&'||c=='|'){ i++; continue; }
        // full-width artifacts Ｏ(EF BC AF) ＆(EF BC 86) ｜(EF BD 9C)
        if(i+2<in.size() && (unsigned char)in[i]==0xEF &&
           (((unsigned char)in[i+1]==0xBC && ((unsigned char)in[i+2]==0xAF || (unsigned char)in[i+2]==0x86)) ||
            ((unsigned char)in[i+1]==0xBD && (unsigned char)in[i+2]==0x9C))){ i+=3; continue; }
        s+=c; i++;
    }
    auto erase_all=[&](const char*sub){ size_t p; while((p=s.find(sub))!=std::string::npos) s.erase(p,strlen(sub)); };
    erase_all("/sil"); erase_all("endofbreak"); erase_all("FFFF");
    // collapse whitespace + trim
    std::string t; t.reserve(s.size());
    bool ws=false;
    for(char c:s){ if(c==' '||c=='\t'||c=='\n'||c=='\r'||c=='\f'||c=='\v'){ ws=true; continue; }
        if(ws && !t.empty()) t+=' '; ws=false; t+=c; }
    return t;
}
// detect_and_fix_hallucination(): repeated word/char-ngram >= max_occurrences -> truncate
// after the 2nd occurrence in the original text. Returns {text, hallucinated}.
static std::pair<std::string,bool> fix_hallucination(const std::string& text, int max_ngram=12, int max_occ=3){
    if(text.empty() || (int)text.size() < max_ngram*2) return {text,false};
    std::string cleaned; cleaned.reserve(text.size());   // roughly \p{P} removal (ASCII punct)
    for(char c:text) if(!ispunct((unsigned char)c)) cleaned+=c;
    auto has_nondigit=[&](const std::string&w){ for(char c:w) if(!isdigit((unsigned char)c)) return true; return false; };
    std::string repeated;
    // word level: same whitespace-separated token repeated >= max_occ times consecutively
    {
        std::vector<std::string> ws; size_t b=0;
        for(size_t i=0;i<=cleaned.size();i++) if(i==cleaned.size()||cleaned[i]==' '||cleaned[i]=='\t'){
            if(i>b)ws.push_back(cleaned.substr(b,i-b)); b=i+1; }
        for(size_t i=0,j;i<ws.size();i++){ for(j=i+1;j<=ws.size();j++){
            if(j<ws.size()){
                std::string a=ws[i],c=ws[j]; for(auto&x:a)x=tolower(x); for(auto&x:c)x=tolower(x);
                if(a==c)continue;
            }
            // run [i,j) of identical tokens ended; python: {max_occurrences-1,} repeats
            // after the first -> trigger at >= max_occurrences total occurrences
            if(j-i>=(size_t)max_occ && has_nondigit(ws[i])){ repeated=ws[i]; break; }
            i=j-1; break;
        } if(!repeated.empty())break; }
    }
    // char-ngram level: L-length block repeated >= max_occ times inside a whitespace-free run
    for(int L=1; repeated.empty() && L<max_ngram; L++){
        for(size_t r0=0; r0<cleaned.size() && repeated.empty();){
            size_t r1=cleaned.find_first_of(" \t",r0); if(r1==std::string::npos)r1=cleaned.size();
            for(size_t i=r0; i+ (size_t)max_occ*L <= r1; i++){
                bool ok=true, digit_only=true;
                for(int k=1;k<max_occ;k++) if(cleaned.compare(i,L,cleaned,i+(size_t)k*L,L)!=0){ok=false;break;}
                if(ok){ for(int k=0;k<L;k++) if(!isdigit((unsigned char)cleaned[i+k])) digit_only=false;
                    if(!digit_only) repeated=cleaned.substr(i,L); break; }
            }
            r0=r1+1;
        }
    }
    if(repeated.empty()) return {text,false};
    size_t p1=text.find(repeated);
    if(p1!=std::string::npos){
        size_t p2=text.find(repeated,p1+repeated.size());
        if(p2!=std::string::npos) return {text.substr(0,p2+repeated.size()),true};
    }
    return {text.substr(0,text.size()/2),true};
}

// ---- streaming mode (--stream): 16k s16le PCM on stdin -> LOCKED/PARTIAL/DONE on stdout ----
// Protocol and thresholds mirror serve_realtime_ws.py's RealtimeASRSession:
//   chunk_ms=960 cadence (first decode at 480ms), partial window capped at 15s of the
//   in-progress VAD segment, locked segments = DynamicStreamingVAD-confirmed segments.
static void emit_line(const char*tag, const std::string&text){
    std::string t=text;
    for(auto&c:t) if(c=='\n'||c=='\r') c=' ';
    if(t.empty()) printf("%s\n",tag);          // DONE, and empty PARTIAL (partial reset)
    else printf("%s %s\n",tag,t.c_str());
    fflush(stdout);
}
static int run_stream(asr_decoder&d, const std::string&vad_path, int vad_maxseg){
    const size_t SR=16000;
    const size_t ANALYSIS=960;            // 60ms VAD/partial analysis quantum
    const size_t FIRST_CHUNK=SR*480/1000, CHUNK=SR*960/1000;   // decode cadence
    const size_t PARTIAL_WIN=SR*15;       // 15s partial window cap
    const size_t MIN_PARTIAL=CHUNK/2, MIN_LOCKED=1600;         // python: chunk//2 / 100ms
    funasr_vad_stream*vs=funasr_vad_stream_open(vad_path.c_str(),vad_maxseg);
    std::vector<float> wav;               // full session audio (64KB/s, fine)
    size_t analyzed=0, last_decode=0; bool first_done=false;
    std::string last_partial;
    for(;;){
        unsigned char blk[16384]; size_t got=fread(blk,1,sizeof blk,stdin);
        bool eof=(got<sizeof blk);
        // s16le -> f32 (carry an odd byte across reads)
        static unsigned char carry=0; static bool have_carry=false;
        size_t i0=0;
        if(have_carry && got>=1){ short v=(short)((blk[0]<<8)|carry); wav.push_back(v/32768.0f); have_carry=false; i0=1; }
        size_t n=(got-i0)/2; size_t base=wav.size(); wav.resize(base+n);
        for(size_t i=0;i<n;i++){ unsigned char b0=blk[i0+2*i], b1=blk[i0+2*i+1];
            wav[base+i]=((short)((b1<<8)|b0))/32768.0f; }
        if(i0+2*n<got){ carry=blk[i0+2*n]; have_carry=true; }
        // quantized analysis checkpoints: VAD emission + locked/partial decode decisions
        // depend only on buffer contents at 60ms-quantized positions, not on pipe read size.
        while(analyzed+ANALYSIS<=wav.size() || (eof && analyzed<wav.size())){
            size_t prev=analyzed;
            analyzed=std::min(analyzed+ANALYSIS, wav.size());
            std::vector<std::pair<int,int>> new_segs;
            if(!funasr_vad_stream_feed(vs,wav.data()+prev,analyzed-prev,new_segs)){fprintf(stderr,"vad failed\n");funasr_vad_stream_close(vs);return 1;}
            for(auto&s:new_segs){                       // DynamicStreamingVAD-confirmed segment -> LOCKED
                size_t s0=(size_t)((int64_t)s.first*SR/1000), s1=(size_t)((int64_t)s.second*SR/1000);
                if(s1>analyzed)s1=analyzed;
                if(s1<=s0||s1-s0<MIN_LOCKED)continue;
                std::string text=clean_asr_text(asr_decode_window(d,wav.data()+s0,s1-s0,512));
                if(text.empty())continue;
                fprintf(stderr,"[stream] locked [%d,%dms] \"%s\"\n",s.first,s.second,text.c_str());
                emit_line("LOCKED",text);
                last_partial.clear();
            }
            // RealtimeASRSession.decode() emulation for the in-progress segment
            if(analyzed<CHUNK)continue;
            if(vs->in_speech_start_ms<0){ last_decode=analyzed;
                if(!last_partial.empty()){ emit_line("PARTIAL",""); last_partial.clear(); }
                continue; }
            size_t threshold=first_done?CHUNK:FIRST_CHUNK;
            if(analyzed-last_decode<threshold)continue;
            size_t start=(size_t)((int64_t)vs->in_speech_start_ms*SR/1000);
            if(start+PARTIAL_WIN<analyzed)start=analyzed-PARTIAL_WIN;   // cap partial window to 15s
            if(analyzed-start<MIN_PARTIAL)continue;
            std::string text=clean_asr_text(asr_decode_window(d,wav.data()+start,analyzed-start,200));
            auto fixed=fix_hallucination(text); text=fixed.first;
            last_decode=analyzed;
            if(text!=last_partial){ emit_line("PARTIAL",text); last_partial=text; }
            if(!text.empty()&&!first_done)first_done=true;
        }
        if(eof)break;
    }
    // STOP: force-decode any trailing in-progress segment (python decode(is_final=True))
    if(vs->in_speech_start_ms>=0){
        size_t s0=(size_t)((int64_t)vs->in_speech_start_ms*SR/1000);
        if(wav.size()>s0&&wav.size()-s0>=MIN_LOCKED){
            std::string text=clean_asr_text(asr_decode_window(d,wav.data()+s0,wav.size()-s0,512));
            if(!text.empty()){ fprintf(stderr,"[stream] locked [%d,%zums] \"%s\"\n",vs->in_speech_start_ms,
                wav.size()*1000/SR,text.c_str()); emit_line("LOCKED",text); }
        }
    }
    emit_line("DONE","");
    funasr_vad_stream_close(vs);
    return 0;
}

int main(int argc,char**argv){
    std::string enc_path,llm_path,wav_path,vad_path,lang; int npred=512; double chunk_sec=0; float rep=1.0f;
    int vad_maxseg=30000; bool stream=false;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--enc")&&i+1<argc)enc_path=argv[++i];
        else if(!strcmp(argv[i],"-m")&&i+1<argc)llm_path=argv[++i];
        else if(!strcmp(argv[i],"-a")&&i+1<argc)wav_path=argv[++i];
        else if(!strcmp(argv[i],"-n")&&i+1<argc)npred=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--chunk")&&i+1<argc)chunk_sec=atof(argv[++i]);
        else if(!strcmp(argv[i],"--vad")&&i+1<argc)vad_path=argv[++i];
        else if(!strcmp(argv[i],"--vad-maxseg")&&i+1<argc)vad_maxseg=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--rep")&&i+1<argc)rep=atof(argv[++i]);
        else if(!strcmp(argv[i],"--lang")&&i+1<argc)lang=argv[++i];
        else if(!strcmp(argv[i],"--stream"))stream=true;
        else {fprintf(stderr,"usage: %s --enc enc.gguf -m llm.gguf (-a audio.wav | --stream) [-n npred] [--chunk sec] [--vad fsmn-vad.gguf [--vad-maxseg ms]] [--lang language]\n",argv[0]);return 1;}
    }
    if(enc_path.empty()||llm_path.empty()||(!stream&&wav_path.empty())){fprintf(stderr,"missing args\n");return 1;}
    if(stream&&vad_path.empty()){fprintf(stderr,"--stream requires --vad fsmn-vad.gguf\n");return 1;}

    std::vector<float> wav;
    if(!stream){
        if(!funasr_load_audio_16k_mono(wav_path.c_str(),wav)){fprintf(stderr,"failed to read audio\n");return 1;}
    }
    int64_t t0=ggml_time_us();

    asr_decoder d;
    if(!load_decoder(enc_path.c_str(),llm_path.c_str(),rep,lang,d)){free_decoder(d);return 1;}

    if(stream){
        int rc=run_stream(d,vad_path,vad_maxseg);
        free_decoder(d);
        return rc;
    }

    // Build the list of [offset,len] windows to transcribe (in samples).
    //   --vad : FSMN-VAD speech segments (single-binary front end, replaces fixed chunking)
    //   --chunk sec : fixed-size chunks ; otherwise the whole file in one window
    std::vector<std::pair<int,int>> wins;   // {sample offset, sample len}
    if(!vad_path.empty()){
        std::vector<std::pair<int,int>> segs; // ms
        if(!funasr_vad_segments(vad_path,wav,vad_maxseg,segs)){fprintf(stderr,"vad failed\n");free_decoder(d);return 1;}
        for(auto&s:segs){ int off=(int)((int64_t)s.first*16000/1000), end=(int)((int64_t)s.second*16000/1000);
            if(end>(int)wav.size())end=wav.size(); if(end-off>0) wins.push_back({off,end-off}); }
        fprintf(stderr,"[vad] %zu segments\n",wins.size());
    } else {
        int chunk_n = chunk_sec > 0 ? std::max(1, (int)(chunk_sec*16000)) : (int)wav.size();
        for(size_t off=0; off<wav.size(); off+=chunk_n) wins.push_back({(int)off,(int)std::min((size_t)chunk_n,wav.size()-off)});
    }
    std::string full;
    for (auto& w : wins) {
        int off = w.first, len = w.second;
        if (len < WINLEN) continue;                    // too short for one frame
        full += asr_decode_window(d, wav.data()+off, len, npred);
    }
    printf("%s\n", full.c_str());
    int64_t t2=ggml_time_us();
    fprintf(stderr,"[done] %.2fs ; chunk=%.0fs\n",(t2-t0)/1e6, chunk_sec);
    free_decoder(d);
    return 0;
}
