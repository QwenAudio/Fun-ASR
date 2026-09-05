// funasr_vad.h — single-header FSMN-VAD for the funasr ggml runtime.
// Offline: funasr_vad_segments(): 16k mono wav -> speech segments [start_ms,end_ms].
// Streaming: funasr_vad_stream_open/feed/close — incremental, irreversible segment
// emission with the FunASR DynamicStreamingVAD schedule (serve_realtime_ws.py).
// Front end (80-mel fbank + LFR m5n1 + CMVN) and FSMN encoder validated bit-exact vs
// PyTorch fsmn-vad; the host state machine reproduces E2EVadModel (chunk-stepped
// classic schedule offline, DynamicStreamingVAD DEFAULT_SILENCE_SCHEDULE when streaming)
// to within 1 frame (10ms) of fsmn-vad.generate on the 184-clip set.
#pragma once
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846   // not guaranteed by <cmath> on MSVC
#endif

namespace funasr_vad_impl {
static const int FS=16000,WINLEN=400,SHIFT=160,NFFT=512,NMEL=80;
static const float PREEMPH=0.97f,LOWF=20.0f,HIGHF=8000.0f;
static inline float melf(float f){return 1127.0f*logf(1.0f+f/700.0f);}
static void fftc(std::vector<float>&re,std::vector<float>&im,int n){for(int i=1,j=0;i<n;i++){int b=n>>1;for(;j&b;b>>=1)j^=b;j^=b;if(i<j){std::swap(re[i],re[j]);std::swap(im[i],im[j]);}}
  for(int len=2;len<=n;len<<=1){double a=-2.0*M_PI/len;float wr=cosf(a),wi=sinf(a);for(int i=0;i<n;i+=len){float cr=1,ci=0;for(int k=0;k<len/2;k++){float ur=re[i+k],ui=im[i+k];
    float vr=re[i+k+len/2]*cr-im[i+k+len/2]*ci,vi=re[i+k+len/2]*ci+im[i+k+len/2]*cr;re[i+k]=ur+vr;im[i+k]=ui+vi;re[i+k+len/2]=ur-vr;im[i+k+len/2]=ui-vi;float nc=cr*wr-ci*wi;ci=cr*wi+ci*wr;cr=nc;}}}}
static std::vector<std::vector<float>> fbank80(std::vector<float> wav){
  for(auto&v:wav)v*=32768.0f; std::vector<float>win(WINLEN);
  for(int i=0;i<WINLEN;i++)win[i]=0.54f-0.46f*cosf(2.0f*M_PI*i/(WINLEN-1));
  const int NB=NFFT/2+1; float bw=(float)FS/NFFT,ml=melf(LOWF),mh=melf(HIGHF),dm=(mh-ml)/(NMEL+1);
  std::vector<std::vector<float>>fb(NMEL,std::vector<float>(NB,0.0f));
  for(int m=0;m<NMEL;m++){float L=ml+m*dm,C=ml+(m+1)*dm,R=ml+(m+2)*dm;for(int k=0;k<NB;k++){float mf=melf(bw*k);if(mf>L&&mf<R)fb[m][k]=mf<=C?(mf-L)/(C-L):(R-mf)/(R-C);}}
  int N=wav.size(),T=(N-WINLEN)/SHIFT+1; if(T<1)T=0; std::vector<std::vector<float>>feat(T,std::vector<float>(NMEL));
  std::vector<float>re(NFFT),im(NFFT),fr(WINLEN);const float fl=1.1920929e-07f;
  for(int t=0;t<T;t++){const float*s=wav.data()+t*SHIFT;double mn=0;for(int i=0;i<WINLEN;i++)mn+=s[i];mn/=WINLEN;
    for(int i=0;i<WINLEN;i++)fr[i]=s[i]-(float)mn;for(int i=WINLEN-1;i>0;i--)fr[i]-=PREEMPH*fr[i-1];fr[0]-=PREEMPH*fr[0];
    for(int i=0;i<NFFT;i++){re[i]=i<WINLEN?fr[i]*win[i]:0.0f;im[i]=0.0f;}fftc(re,im,NFFT);
    for(int m=0;m<NMEL;m++){float e=0;for(int k=0;k<NB;k++)if(fb[m][k]>0)e+=fb[m][k]*(re[k]*re[k]+im[k]*im[k]);feat[t][m]=logf(e>fl?e:fl);}}
  return feat;
}
// LFR over a window of features with explicit global-edge handling.
// Frames are stacked [i - (m-1)/2 .. i + m/2], clamped to [0, T-1] by edge replication,
// matching the padding of the original batch lfr() below.
static std::vector<std::vector<float>> lfr(const std::vector<std::vector<float>>&feat,int m,int n,int&T_out){
  int T=feat.size(); if(T<1){T_out=0;return {};}     // empty (audio shorter than one frame)
  int D=NMEL,pad=(m-1)/2; int Tl=(T+n-1)/n;
  std::vector<std::vector<float>> pf; pf.reserve(T+pad+m);
  for(int i=0;i<pad;i++)pf.push_back(feat[0]);
  for(int t=0;t<T;t++)pf.push_back(feat[t]);
  while((int)pf.size()<(Tl-1)*n+m)pf.push_back(feat[T-1]);
  std::vector<float> flat((size_t)Tl*m*D);
  for(int i=0;i<Tl;i++)for(int j=0;j<m;j++)memcpy(&flat[((size_t)i*m+j)*D],pf[i*n+j].data(),D*sizeof(float));
  T_out=Tl;
  // keep the historical [T x m*D] layout via re-wrap so callers stay unchanged
  std::vector<std::vector<float>> rows(Tl,std::vector<float>(m*D));
  for(int i=0;i<Tl;i++)memcpy(rows[i].data(),&flat[(size_t)i*m*D],(size_t)m*D*sizeof(float));
  return rows;
}
struct vad{ggml_context*ctx=nullptr;std::map<std::string,ggml_tensor*>t;
  ggml_tensor*g(const std::string&n){auto it=t.find(n);if(it==t.end()){fprintf(stderr,"vad: missing %s\n",n.c_str());return nullptr;}return it->second;}};
static ggml_tensor* lin(ggml_context*c,ggml_tensor*w,ggml_tensor*b,ggml_tensor*x){auto y=ggml_mul_mat(c,w,x);return b?ggml_add(c,y,b):y;}

// ---- model handle ----
struct vad_model{
  vad m; gguf_context*gg=nullptr;
  int idim=400,pd=128,nl=4,lorder=20,od=248,lm=5,ln=1;
  bool ok=false;
};
static bool funasr_vad_load(const std::string& gguf_path, vad_model& vm){
  gguf_init_params ip={false,&vm.m.ctx}; vm.gg=gguf_init_from_file(gguf_path.c_str(),ip);
  if(!vm.gg){fprintf(stderr,"vad: cannot load %s\n",gguf_path.c_str());return false;}
  auto rd=[&](const char*k,int d){int i=gguf_find_key(vm.gg,k);return i<0?d:(int)gguf_get_val_u32(vm.gg,i);};
  vm.idim=rd("vad.input_dim",400);vm.pd=rd("vad.proj_dim",128);vm.nl=rd("vad.fsmn_layers",4);
  vm.lorder=rd("vad.lorder",20);vm.od=rd("vad.output_dim",248);vm.lm=rd("vad.lfr_m",5);vm.ln=rd("vad.lfr_n",1);
  for(int i=0;i<gguf_get_n_tensors(vm.gg);i++){const char*nm=gguf_get_tensor_name(vm.gg,i);vm.m.t[nm]=ggml_get_tensor(vm.m.ctx,nm);}
  auto need=[&](const std::string&n){ return vm.m.g(n)!=nullptr; };
  vm.ok = need("cmvn.shift")&&need("cmvn.scale")&&need("encoder.in_linear1.linear.weight")
        &&need("encoder.in_linear2.linear.weight")&&need("encoder.out_linear1.linear.weight")
        &&need("encoder.out_linear2.linear.weight");
  for(int i=0;i<vm.nl&&vm.ok;i++){std::string p="encoder.fsmn."+std::to_string(i)+".";
    vm.ok=need(p+"linear.linear.weight")&&need(p+"fsmn_block.conv_left.weight")&&need(p+"affine.linear.weight");}
  if(!vm.ok)fprintf(stderr,"vad: gguf missing required tensors\n");
  return vm.ok;
}
static void funasr_vad_unload(vad_model& vm){
  if(vm.gg)gguf_free(vm.gg);
  if(vm.m.ctx)ggml_free(vm.m.ctx);
  vm=vad_model{};
}
// scores[lfr_t] = silence prob for LFR frame lfr_t; feats are raw (pre-CMVN) [T x idim], row-major
static bool funasr_vad_scores(vad_model& vm, const std::vector<float>& feats_in, int T,
                              std::vector<float>& sil, int nthreads){
  sil.assign(T,1.0f);
  if(T<1)return true;
  std::vector<float> feats=feats_in;
  float*shift=(float*)vm.m.g("cmvn.shift")->data,*scale=(float*)vm.m.g("cmvn.scale")->data;
  for(int t=0;t<T;t++)for(int d=0;d<vm.idim;d++)feats[(size_t)t*vm.idim+d]=(feats[(size_t)t*vm.idim+d]+shift[d])*scale[d];
  auto&v=vm.m;
  ggml_backend_t be=ggml_backend_cpu_init();
  // no_alloc=true -> ctx holds only tensor/graph metadata (the real compute buffer is
  // allocated by gallocr below), so a few MB is plenty regardless of clip length.
  ggml_init_params cp={(size_t)16*1024*1024,nullptr,true}; ggml_context*c=ggml_init(cp);
  ggml_tensor*x=ggml_new_tensor_2d(c,GGML_TYPE_F32,vm.idim,T); ggml_set_input(x);
  ggml_tensor*h=lin(c,v.g("encoder.in_linear1.linear.weight"),v.g("encoder.in_linear1.linear.bias"),x);
  h=lin(c,v.g("encoder.in_linear2.linear.weight"),v.g("encoder.in_linear2.linear.bias"),h); h=ggml_relu(c,h);
  for(int i=0;i<vm.nl;i++){std::string p="encoder.fsmn."+std::to_string(i)+".";
    ggml_tensor*z=ggml_mul_mat(c,v.g(p+"linear.linear.weight"),h);
    ggml_tensor*fk=v.g(p+"fsmn_block.conv_left.weight"); ggml_tensor*zp=ggml_pad_ext(c,z,0,0,vm.lorder-1,0,0,0,0,0); ggml_tensor*acc=z;
    // sl is a full-row slice of the contiguous padded tensor -> already contiguous, no ggml_cont needed
    for(int j=0;j<vm.lorder;j++){auto sl=ggml_view_2d(c,zp,vm.pd,T,zp->nb[1],(size_t)j*zp->nb[1]);auto wj=ggml_view_1d(c,fk,vm.pd,(size_t)j*fk->nb[1]);acc=ggml_add(c,acc,ggml_mul(c,sl,wj));}
    ggml_tensor*a=lin(c,v.g(p+"affine.linear.weight"),v.g(p+"affine.linear.bias"),acc); h=ggml_relu(c,a);}
  h=lin(c,v.g("encoder.out_linear1.linear.weight"),v.g("encoder.out_linear1.linear.bias"),h);
  h=lin(c,v.g("encoder.out_linear2.linear.weight"),v.g("encoder.out_linear2.linear.bias"),h);
  h=ggml_soft_max(c,h); ggml_set_output(h);
  ggml_cgraph*gf=ggml_new_graph(c); ggml_build_forward_expand(gf,h);
  ggml_gallocr_t ga=ggml_gallocr_new(ggml_backend_cpu_buffer_type()); ggml_gallocr_alloc_graph(ga,gf);
  ggml_backend_tensor_set(x,feats.data(),0,ggml_nbytes(x)); ggml_backend_cpu_set_n_threads(be,nthreads);
  bool ok=ggml_backend_graph_compute(be,gf)==GGML_STATUS_SUCCESS;
  std::vector<float> sc((size_t)vm.od*T); if(ok)ggml_backend_tensor_get(h,sc.data(),0,ggml_nbytes(h));
  ggml_gallocr_free(ga);ggml_free(c);ggml_backend_free(be);
  if(!ok)return false;
  for(int t=0;t<T;t++)sil[t]=sc[(size_t)t*vm.od+0];
  return true;
}

// ---- E2EVadModel state machine (host) ----
// Stepped per LFR frame (10ms). dynamic_schedule=false reproduces the offline
// fsmn-vad.generate behaviour (60s-chunk-stepped schedule); true follows
// DynamicStreamingVAD's DEFAULT_SILENCE_SCHEDULE keyed on frames since the last emit.
struct vad_machine{
  static const int FR=10;                 // ms per frame (frame_in_ms)
  int T_ms=0;                             // frames stepped so far (== frame count)
  int max_seg=60000/FR;                   // max_single_segment frames
  bool dynamic=false;
  std::vector<int> wbuf; int win=20;      // window_size_ms 200 / FR
  int wpos=0,wsum=0,pre=0;
  int st=0,cstart=-1,csil=0,prev_end=0;
  int acc=0,insp=0,last_emit=0;
  vad_machine(int max_seg_ms=60000,bool dynamic_schedule=false){
    max_seg=(max_seg_ms>0?max_seg_ms:60000)/FR; dynamic=dynamic_schedule;
    wbuf.assign(win,0);
  }
  int max_end_sil(int t)const{
    int ms;
    if(!dynamic){
      int s; if(acc<=10000)s=2000; else if(acc<=20000)s=1000; else if(acc<=30000)s=800;
      else if(acc<=40000)s=600; else if(acc<=50000)s=400; else if(acc<=60000)s=200; else s=100;
      ms=s;
    } else {
      int a=(t-last_emit)*FR;
      int s; if(a<=5000)s=2000; else if(a<=10000)s=1500; else if(a<=15000)s=1000;
      else if(a<=30000)s=800; else if(a<=45000)s=400; else s=100;
      ms=s;
    }
    ms-=150; if(ms<0)ms=0; return ms/FR;
  }
  static int end_lookback(int mes){
    const int lookahead_end=100/FR;       // lookahead_time_end_point 100/10 = 10 (do_extend)
    int e=mes-lookahead_end-1; return e<0?0:e;
  }
  void reset(){ std::fill(wbuf.begin(),wbuf.end(),0); wpos=0; wsum=0; pre=0; csil=0; st=0; cstart=-1; acc=0; insp=0; }
  // step one frame; appends frames-delimited segments to segs when confirmed
  void step(int t, float sil, std::vector<std::pair<int,int>>& segs){
    const int s2s=15, sp2s=15;            // sil_to_speech / speech_to_sil thres (150/10)
    if(!dynamic && t>0 && t%(60000/FR)==0){ if(st==1||insp){acc+=60000; insp=1;} }
    int mes=max_end_sil(t);
    int elb=end_lookback(mes);
    int fs = ((1.0f-sil) >= sil + 0.5f) ? 1 : 0;  // speech_noise_thres=0.5
    wsum -= wbuf[wpos]; wsum += fs; wbuf[wpos]=fs; wpos=(wpos+1)%win;
    int ch;
    if(pre==0 && wsum>=s2s){pre=1; ch=3;}
    else if(pre==1 && wsum<=sp2s){pre=0; ch=1;}
    else ch = pre==0?0:2;
    auto emit=[&](int s,int e,int T){
      if(s<prev_end)s=prev_end; if(s<0)s=0; if(e>T)e=T;
      if(e>s){segs.push_back({s,e}); prev_end=e; last_emit=e;}
    };
    const int start_lookback = win + 200/FR;  // LatencyFrmNumAtStartPoint = 40
    if(ch==3){ csil=0;
      if(st==0){ cstart=t-start_lookback; if(cstart<prev_end)cstart=prev_end; if(cstart<0)cstart=0; st=1; }
      else if(st==1 && t-cstart+1>max_seg){ emit(cstart,t,T_ms+1); reset(); }
    } else if(ch==1||ch==2){ csil=0;
      if(st==1 && t-cstart+1>max_seg){ emit(cstart,t,T_ms+1); reset(); }
    } else { csil++;
      if(st==1){
        if(csil>=mes){ emit(cstart, t-elb, T_ms+1); reset(); }
        else if(t-cstart+1>max_seg){ emit(cstart,t,T_ms+1); reset(); }
      }
    }
    T_ms=t+1;
  }
};
} // namespace funasr_vad_impl

// Full-buffer analysis. dynamic_schedule=false keeps the classic offline behaviour
// (60s-chunk-stepped silence schedule, trailing in-progress speech flushed into segs).
// dynamic_schedule=true follows DynamicStreamingVAD (serve_realtime_ws.py), keyed on
// frames elapsed since the last emitted segment, and reports the in-progress segment
// start in in_speech_start_ms_out (-1 when the trailing frames are not in speech).
inline bool funasr_vad_analyze(const std::string& gguf_path, const std::vector<float>& wav,
                               int max_seg_ms, bool dynamic_schedule,
                               std::vector<std::pair<int,int>>& segs,
                               int* in_speech_start_ms_out, int nthreads=8){
  using namespace funasr_vad_impl;
  segs.clear();
  if(in_speech_start_ms_out)*in_speech_start_ms_out=-1;
  vad_model vm;
  if(!funasr_vad_load(gguf_path,vm)){funasr_vad_unload(vm);return false;}
  auto feat=fbank80(wav); int T=0; auto rows=lfr(feat,vm.lm,vm.ln,T);   // [T,400]
  std::vector<float> feats((size_t)T*vm.idim);
  for(int t=0;t<T;t++)memcpy(&feats[(size_t)t*vm.idim],rows[t].data(),(size_t)vm.idim*sizeof(float));
  std::vector<float> sil;
  bool ok=funasr_vad_scores(vm,feats,T,sil,nthreads);
  funasr_vad_unload(vm);
  if(!ok)return false;
  std::vector<std::pair<int,int>> segs_fr;
  vad_machine mach(max_seg_ms,dynamic_schedule);
  for(int t=0;t<T;t++)mach.step(t,sil[t],segs_fr);
  if(in_speech_start_ms_out){
    *in_speech_start_ms_out = (mach.st==1) ? mach.cstart*vad_machine::FR : -1;
  } else if(mach.st==1){
    int s=mach.cstart; if(s<mach.prev_end)s=mach.prev_end; if(s<0)s=0;
    if(T>s)segs_fr.push_back({s,T});
  }
  for(auto&s:segs_fr){ s.first*=vad_machine::FR; s.second*=vad_machine::FR; segs.push_back(s); }
  return true;
}

// Backwards-compatible offline entry point (classic schedule, trailing speech flushed).
inline bool funasr_vad_segments(const std::string& gguf_path, const std::vector<float>& wav,
                                int max_seg_ms, std::vector<std::pair<int,int>>& segs, int nthreads=8){
  return funasr_vad_analyze(gguf_path, wav, max_seg_ms, false, segs, nullptr, nthreads);
}

// ---- streaming session: irreversible DynamicStreamingVAD-style segment emission ----
// Scores are computed incrementally: the FSMN stack has lorder-1 left context per layer
// (nl layers -> (lorder-1)*nl frames) plus the 2-frame LFR left context, so scoring a
// new block only needs a bounded rewind; the last 2 frames are unstable (LFR needs 2
// future frames) and are recomputed on the next feed. The state machine steps only over
// stable frames, which makes emissions identical to a full re-analysis.
struct funasr_vad_stream {
  std::string gguf_path;
  funasr_vad_impl::vad_model vm;
  std::vector<float> wav;                 // full session buffer, 16k mono [-1,1]
  std::vector<float> sil;                 // stable-frame silence scores
  size_t stable_frames=0;                 // frames scored+stepped so far
  funasr_vad_impl::vad_machine mach;
  std::vector<std::pair<int,int>> emitted;// confirmed segments, ms
  int in_speech_start_ms=-1;
  int nthreads=8;
  funasr_vad_stream(const char*path,int max_seg_ms):mach(max_seg_ms>0?max_seg_ms:60000,true){
    if(path)gguf_path=path;
  }
};
inline void funasr_vad_stream_close(funasr_vad_stream* s);
inline funasr_vad_stream* funasr_vad_stream_open(const char* gguf_path, int max_seg_ms){
  auto* s=new funasr_vad_stream(gguf_path,max_seg_ms);
  if(!funasr_vad_impl::funasr_vad_load(s->gguf_path,s->vm)){funasr_vad_stream_close(s);return nullptr;}
  return s;
}
// Append samples; returns (via new_segs) segments confirmed since the previous call.
inline bool funasr_vad_stream_feed(funasr_vad_stream* s, const float* pcm, size_t n,
                                   std::vector<std::pair<int,int>>& new_segs){
  using namespace funasr_vad_impl;
  new_segs.clear();
  if(!s||!s->vm.ok)return false;
  if(pcm&&n)s->wav.insert(s->wav.end(),pcm,pcm+n);
  const int N=(int)s->wav.size();
  const int Tfb = N>=WINLEN ? (N-WINLEN)/SHIFT+1 : 0;    // fbank frames available
  const int target = Tfb>=2 ? Tfb-2 : 0;                 // stable LFR frames
  const int lb_frames = (s->vm.nl)*(s->vm.lorder-1);     // FSMN left context (e.g. 76)
  if(target>(int)s->stable_frames){
    int first=(int)s->stable_frames;
    int w0 = first==0 ? 0 : first-lb_frames; if(w0<0)w0=0;
    int w1 = target;
    // score LFR frames [w0,w1): fbank frames [f0 .. f1) with LFR edge context
    int f0 = w0-2; if(f0<0)f0=0;
    int f1 = w1+2; if(f1>Tfb)f1=Tfb;
    if(w1>w0&&f1>f0){
      // slice audio aligned to fbank frame boundaries (frames are independent)
      const float*base=s->wav.data()+(size_t)f0*SHIFT;
      int nsamp=(int)std::min((int64_t)N-(int64_t)f0*SHIFT,(int64_t)(f1-1)*SHIFT+WINLEN-(int64_t)f0*SHIFT);
      std::vector<float> slice(base,base+nsamp);
      auto feat=fbank80(slice); int Tl=0; auto rows=lfr(feat,s->vm.lm,s->vm.ln,Tl);
      // global LFR index i in [w0,w1) must resolve to local row (i - f0): true because
      // lfr() over the slice of fbank frames [f0..) reproduces the same clamping only at
      // global edges; for w0>0 no clamping occurs inside the window.
      // NOTE: lfr()/fbank80() above keep the historical interface; the local row count
      // covers [f0, f0+Tl).
      std::vector<float> feats((size_t)(w1-w0)*s->vm.idim);
      bool ok=true;
      for(int i=w0;i<w1&&ok;i++){
        int li=i-f0;
        if(li<0||li>=Tl)ok=false;   // slice too short (audio truncated mid-frame)
        else memcpy(&feats[(size_t)(i-w0)*s->vm.idim],rows[li].data(),(size_t)s->vm.idim*sizeof(float));
      }
      std::vector<float> sil_w;
      ok=ok&&funasr_vad_scores(s->vm,feats,w1-w0,sil_w,s->nthreads);
      if(!ok)return false;
      // outputs are valid from w0 (edge-padded conv) when w0==0, else after the
      // left-context warmup; keep frames [first, w1)
      int keep0 = std::max(w0 + (w0>0?lb_frames:0), first);
      for(int i=keep0;i<w1;i++)s->sil.push_back(sil_w[i-w0]);
      if((int)s->sil.size()<w1 && keep0<w1)s->sil.resize(w1);  // keep indices aligned
      std::vector<std::pair<int,int>> fr;
      for(int t=first;t<w1;t++)s->mach.step(t,s->sil[t],fr);
      s->stable_frames=w1;
      for(auto&p:fr){ p.first*=vad_machine::FR; p.second*=vad_machine::FR; new_segs.push_back(p); s->emitted.push_back({p.first,p.second}); }
      s->in_speech_start_ms = (s->mach.st==1) ? s->mach.cstart*vad_machine::FR : -1;
    }
  }
  return true;
}
inline void funasr_vad_stream_close(funasr_vad_stream* s){
  if(!s)return;
  funasr_vad_impl::funasr_vad_unload(s->vm);
  delete s;
}
