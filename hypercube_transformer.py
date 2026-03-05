import numpy as np
from dataclasses import dataclass

# HYPERCUBE TRANSFORMER - 6D State Space Navigation
# Circumpunct Framework - Ashman Roonz, 2026

@dataclass
class HypercubeConfig:
    vocab_size: int = 256
    seq_len: int = 64
    d_model: int = 192
    n_layers: int = 6
    d_vertex: int = 32
    temperature: float = 1.0
    @property
    def d_aperture(self): return self.d_model // 3
    @property
    def d_field(self): return self.d_model // 3
    @property
    def d_boundary(self): return self.d_model // 3

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
def gelu(x): return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0/np.pi) * (x + 0.044715*x**3)))
def smax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)
def lnorm(x, g, b, eps=1e-5):
    m = np.mean(x, axis=-1, keepdims=True)
    v = np.var(x, axis=-1, keepdims=True)
    return g * (x - m) / np.sqrt(v + eps) + b

class Hypercube6D:
    NAMES = ["b1","b2","c1","c2","r1","r2"]
    def __init__(self):
        self.vertices = np.array([[(v>>i)&1 for i in range(6)] for v in range(64)], dtype=np.float64)
        self.adjacency = np.zeros((64,64))
        for i in range(64):
            for d in range(6): self.adjacency[i, i^(1<<d)] = 1.0
        self.hamming = np.zeros((64,64))
        for i in range(64):
            for j in range(64): self.hamming[i,j] = bin(i^j).count("1")
        self.laplacian = np.diag(self.adjacency.sum(1)) - self.adjacency
        self.openness = {}
        for k in range(7): self.openness[k] = [v for v in range(64) if bin(v).count("1")==k]
        self.subcubes = {
            "aperture": {"dims":[0,1], "verts":[v for v in range(64) if all((v>>d)&1==0 for d in [2,3,4,5])], "label":"Pure presence"},
            "boundary": {"dims":[2,3], "verts":[v for v in range(64) if all((v>>d)&1==0 for d in [0,1,4,5])], "label":"Pure boundary"},
            "resonant": {"dims":[4,5], "verts":[v for v in range(64) if all((v>>d)&1==0 for d in [0,1,2,3])], "label":"Pure resonance"},
            "functional": {"dims":[0,1,2,3], "verts":[v for v in range(64) if all((v>>d)&1==0 for d in [4,5])], "label":"Functional love"},
            "resonant_love": {"dims":[0,1,4,5], "verts":[v for v in range(64) if all((v>>d)&1==0 for d in [2,3])], "label":"Resonant love"},
        }
    def neighbors(self, v): return [v^(1<<d) for d in range(6)]
    def spectral_emb(self, n=6):
        _, ev = np.linalg.eigh(self.laplacian)
        return ev[:, 1:n+1]

class Embedding:
    def __init__(self, c):
        s=0.02; da,df,db = c.d_aperture, c.d_field, c.d_boundary
        self.Wa = np.random.randn(c.vocab_size, da)*s
        self.Wf = np.random.randn(c.vocab_size, df)*s
        self.Wb = np.random.randn(c.vocab_size, db)*s
        pe = np.zeros((c.seq_len, db))
        pos = np.arange(c.seq_len)[:,None]
        div = np.exp(np.arange(0,db,2)*-(np.log(10000.0)/db))
        pe[:,0::2]=np.sin(pos*div); pe[:,1::2]=np.cos(pos*div)
        self.pe=pe; self.lg=np.ones(c.d_model); self.lb=np.zeros(c.d_model)
    def forward(self, ids):
        a=self.Wa[ids]; f=self.Wf[ids]; b=self.Wb[ids]+self.pe[:ids.shape[1]]
        return a, f, b
    def compose(self, a, f, b):
        return lnorm(np.concatenate([a,f,b], axis=-1), self.lg, self.lb)

class HypercubeAttn:
    def __init__(self, c, cube):
        self.c=c; self.cube=cube; s=0.02; d=c.d_model; dv=c.d_vertex
        self.Wq=np.random.randn(d,dv)*s; self.Wk=np.random.randn(d,dv)*s
        self.vemb=np.random.randn(64,dv)*s
        self.Wv=np.random.randn(d,d)*s
        self.vgates=np.random.randn(64,d)*s
        self.adj_bias=cube.adjacency*0.5
        self.spec=cube.spectral_emb(6); self.Wsp=np.random.randn(6,dv)*s
        self.Wam=np.random.randn(d,d//3)*s
        self.Wbm=np.random.randn(d,d//3)*s
        self.Wrm=np.random.randn(d,d//3)*s
        self.Wo=np.random.randn(d,d)*s; self.bo=np.zeros(d)
        self.lg=np.ones(d); self.lb=np.zeros(d)

    def forward(self, ap, fi, bo, x, mask=None):
        B,S,d=x.shape
        q=x@self.Wq; k=x@self.Wk
        inter=q[:,:,np.newaxis,:]*k[:,np.newaxis,:,:]
        vs=inter@self.vemb.T
        sf=self.spec@self.Wsp; sb=inter@sf.T
        vs=vs+0.1*sb+0.1*(vs@self.adj_bias)
        pi=smax(vs/self.c.temperature, axis=-1)
        if mask is not None: pi=pi*mask[:,:,:,np.newaxis]
        V=x@self.Wv; gates=sigmoid(self.vgates)
        wp=np.einsum("bsjv,vd->bsjd",pi,gates)
        ws=wp.sum(axis=2,keepdims=True)+1e-8
        out=np.einsum("bijd,bjd->bid",wp/ws,V)
        av=[v for v in range(64) if(v&0b000011)]
        bv=[v for v in range(64) if(v&0b001100)]
        rv=[v for v in range(64) if(v&0b110000)]
        def mattn(verts, Wm):
            pm=pi[:,:,:,verts].sum(axis=-1)
            pm=pm/(pm.sum(axis=-1,keepdims=True)+1e-8)
            return np.einsum("bij,bjd->bid",pm,x@Wm)
        mo=np.concatenate([mattn(av,self.Wam),mattn(bv,self.Wbm),mattn(rv,self.Wrm)],axis=-1)
        r=(out+mo)@self.Wo+self.bo
        return lnorm(r,self.lg,self.lb), pi

class FFN:
    def __init__(self, c):
        d=c.d_model; ff=d*4; s=0.02
        self.W1=np.random.randn(d,ff)*s; self.b1=np.zeros(ff)
        self.W2=np.random.randn(ff,d)*s; self.b2=np.zeros(d)
        self.lg=np.ones(d); self.lb=np.zeros(d)
    def forward(self, x):
        return lnorm(gelu(x@self.W1+self.b1)@self.W2+self.b2, self.lg, self.lb)

class Block:
    def __init__(self, c, cube):
        self.attn=HypercubeAttn(c,cube); self.ff=FFN(c)
        d=c.d_model; s=0.02
        self.Wda=np.random.randn(d,c.d_aperture)*s
        self.Wdf=np.random.randn(d,c.d_field)*s
        self.Wdb=np.random.randn(d,c.d_boundary)*s
    def forward(self, x, mask=None):
        a,f,b=x@self.Wda, x@self.Wdf, x@self.Wdb
        ao,pi=self.attn.forward(a,f,b,x,mask)
        x=x+ao; x=x+self.ff.forward(x)
        return x, pi

class HypercubeTransformer:
    def __init__(self, c):
        self.c=c; self.cube=Hypercube6D()
        self.emb=Embedding(c)
        self.blocks=[Block(c,self.cube) for _ in range(c.n_layers)]
        s=0.02
        self.Wo=np.random.randn(c.d_model,c.vocab_size)*s
        self.bo=np.zeros(c.vocab_size)
        self.lg=np.ones(c.d_model); self.lb=np.zeros(c.d_model)
    def forward(self, ids):
        B,S=ids.shape
        a,f,b=self.emb.forward(ids); x=self.emb.compose(a,f,b)
        mask=np.broadcast_to(np.tril(np.ones((S,S))),(B,S,S))
        pis=[]
        for bl in self.blocks: x,pi=bl.forward(x,mask); pis.append(pi)
        x=lnorm(x,self.lg,self.lb)
        return x@self.Wo+self.bo, pis

def analyze(pi, cube):
    avg=pi.mean(axis=(0,1,2))
    op=np.zeros(7)
    for k in range(7): op[k]=avg[cube.openness[k]].sum()
    pascal=np.array([1,6,15,20,15,6,1])/64
    ent=-np.sum(avg[avg>0]*np.log2(avg[avg>0]))
    ac=0.0; ne=0
    for v1 in range(64):
        for v2 in cube.neighbors(v1):
            if v2>v1: ac+=avg[v1]*avg[v2]; ne+=1
    ac/=ne; rac=(1/64)**2
    def mm(dims):
        vs=[v for v in range(64) if any((v>>d)&1 for d in dims)]
        return avg[vs].sum()
    return dict(avg=avg, openness=op, pascal=pascal, entropy=ent,
                locality=ac/(rac+1e-10),
                modes=dict(aperture=mm([0,1]),boundary=mm([2,3]),resonant=mm([4,5])),
                top10=np.argsort(avg)[::-1][:10])

def demo():
    print("="*70)
    print("  HYPERCUBE TRANSFORMER -- 6D State Space Navigation")
    print("  64 vertices x 192 edges x infinite relational depth")
    print("="*70)
    cube=Hypercube6D()
    print("\nS1 -- The 6D Hypercube")
    print("    Vertices:  64 (= 2^6 relational states)")
    print(f"    Edges:     {int(cube.adjacency.sum()//2)} (single gate flips)")
    print("    Degree:    6 (each state has 6 neighbors)")
    print("    Diameter:  6 (max distance = all gates flip)")
    print("\n    Openness layers (Pascal row 6):")
    for k in range(7):
        n=len(cube.openness[k])
        print(f"      k={k}: {n:>2d} vertices  " + "#"*n + "."*(20-n))
    evals=np.linalg.eigvalsh(cube.laplacian)
    ue=np.unique(np.round(evals,4))
    print("\n    Laplacian spectrum:")
    for e in ue:
        m=int(np.sum(np.abs(evals-e)<0.01))
        tag=" <-- Pascal!" if m in [1,6,15,20] else ""
        print(f"      lambda={e:>5.1f}  mult={m:>2d}{tag}")
    print("\n    Subcubes:")
    for nm,info in cube.subcubes.items():
        vkey = "verts"
        print(f"      {nm:18s}: {len(info[vkey]):>2d} verts -- {info['label']}")
    cfg=HypercubeConfig(vocab_size=256,seq_len=32,d_model=192,n_layers=6,d_vertex=32)
    print("\nS2 -- Architecture")
    print(f"    d_model:  {cfg.d_model} = {cfg.d_aperture}(a)+{cfg.d_field}(f)+{cfg.d_boundary}(b)")
    print(f"    n_layers: {cfg.n_layers}")
    print(f"    d_vertex: {cfg.d_vertex}")
    np.random.seed(42)
    model=HypercubeTransformer(cfg)
    B,S=2,12
    ids=np.random.randint(0,cfg.vocab_size,(B,S))
    print("\nS3 -- Forward Pass")
    print(f"    Input:  ({B}, {S})")
    logits,pis=model.forward(ids)
    print(f"    Output: {logits.shape}")
    print(f"    Vertex dists: {len(pis)} layers x {pis[0].shape}")
    print("\nS4 -- Hypercube Attention Analysis")
    for li,pi in enumerate(pis):
        a=analyze(pi,cube)
        print(f"\n    Layer {li+1}:")
        ekey='entropy'; print(f"      Entropy: {a[ekey]:.2f} / 6.0 bits")
        print(f"      Locality ratio: {a["locality"]:.2f}x random")
        print("      Mode mass:")
        for mode,mass in a["modes"].items():
            bar="#"*int(mass*30)+"."*(30-int(mass*30))
            print(f"        {mode:12s} {bar} {mass:.3f}")
        print("      Openness:")
        for k in range(7):
            obs=a["openness"][k]; exp=a["pascal"][k]
            bar="#"*int(obs*40)+"."*(40-int(obs*40))
            print(f"        k={k}: {bar} {obs:.3f} (Pascal: {exp:.3f})")
    fa=analyze(pis[-1],cube)
    print("\nS5 -- Top Relational States (Final Layer)")
    nms=["b1","b2","c1","c2","r1","r2"]
    for rank,v in enumerate(fa["top10"]):
        bits=format(v,"06b")
        sig="("+",".join(bits)+")"
        p=fa["avg"][v]
        og=[nms[i] for i in range(6) if bits[i]=="1"]
        os="+".join(og) if og else "NULL"
        print(f"    #{rank+1:>2d}  state={v:>2d}  sigma={sig}  prob={p:.4f}  [{os}]")
    print()
    print("  VP-Transformer:        6 independent gates -> aggregate")
    print("  Hypercube-Transformer: 64 states as GRAPH -> navigate")
    print()
    print("  The 64 states are not just a set -- they are a GRAPH.")
    print("  The hypercube IS the topology of possible relationships.")
    print("  Attention is navigation through relational space.")
    print("="*70)
    return model, logits, pis, cube

if __name__=="__main__":
    demo()
