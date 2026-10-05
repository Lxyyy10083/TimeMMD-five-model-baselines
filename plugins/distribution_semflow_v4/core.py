"""Conditional trajectory density, semantic trust and distribution-derived decision."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


def positions(length, width, device, start=0):
    t=torch.arange(start,start+length,device=device).float()[:,None]
    f=torch.exp(-math.log(10000)*torch.arange(0,width,2,device=device).float()/width)
    p=torch.zeros(length,width,device=device)
    p[:,0::2]=torch.sin(t*f);p[:,1::2]=torch.cos(t*f[:p[:,1::2].shape[1]])
    return p


class EventMemory(nn.Module):
    def __init__(self,width=64,text_dim=768):
        super().__init__();self.width=width
        self.numeric=nn.GRU(2,width,batch_first=True)
        self.text=nn.Sequential(nn.Linear(text_dim,width),nn.LayerNorm(width),nn.SiLU())
        self.query=nn.Linear(width,width);self.key=nn.Linear(width,width);self.value=nn.Linear(width,width)
        self.relevance=nn.Sequential(nn.Linear(3*width,width),nn.SiLU(),nn.Linear(width,1))
        self.null_bias=nn.Parameter(torch.tensor(1.0))

    def forward(self,history,text,mask):
        self.level=history.mean(1,keepdim=True).detach()
        self.scale=history.std(1,keepdim=True,unbiased=False).clamp(0.1,2.0).detach()
        x=(history-self.level)/self.scale
        diff=F.pad(x[:,1:]-x[:,:-1],(0,0,1,0))
        states,_=self.numeric(torch.cat((x,diff),-1))
        self.valid=mask.bool()
        semantic=self.text(torch.where(self.valid[...,None],text,torch.zeros_like(text)))
        self.gate=torch.sigmoid(self.relevance(torch.cat((states,semantic,states*semantic),-1)))
        self.numeric_states=states;self.event_states=semantic*self.gate
        self.position=positions(history.shape[1],self.width,history.device)
        self.last_observation=history[:,-1:, :]
        self.modulation_cost=history.new_zeros(())
        return states

    def lookup(self,query):
        keys=self.key(self.event_states+self.position[None])
        scores=self.query(query)@keys.transpose(-1,-2)/math.sqrt(self.width)
        scores=scores.masked_fill(~self.valid[:,None],-1e4)
        # Explicit null event: attention need not select any available text.
        count=self.valid.sum(-1).clamp_min(1).to(scores.dtype)
        null=(count.log()+self.null_bias)[:,None,None].expand(-1,query.shape[1],1)
        weights=torch.softmax(torch.cat((scores,null),-1),-1)[...,:-1]*self.valid[:,None]
        values=self.value(self.event_states)*self.valid[...,None]
        return weights@values,weights


class InternalFusion(nn.Module):
    def __init__(self,feature_dim,width=64,complex_features=False):
        super().__init__();self.complex_features=complex_features;self.feature_dim=feature_dim
        incoming=2*feature_dim if complex_features else feature_dim
        self.down=nn.Linear(incoming,width);self.up=nn.Linear(width,incoming)
        self.modulate=nn.Sequential(nn.Linear(2*width,width),nn.SiLU())
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
        self.strength=nn.Parameter(torch.tensor(-2.0))

    def forward(self,features,memory):
        real=torch.cat((features.real,features.imag),-1) if self.complex_features else features
        shape=real.shape;tokens=real.reshape(shape[0],-1,shape[-1]);q=self.down(tokens)
        if self.complex_features:
            # Frequency-index conditioning, rather than treating bins as timestamps.
            q=q+positions(q.shape[1],q.shape[-1],q.device)[None]
        evidence,attention=memory.lookup(q)
        mixed=self.modulate(torch.cat((F.layer_norm(q,(q.shape[-1],)),evidence),-1))
        update=torch.tanh(self.up(mixed*torch.tanh(evidence)))
        trust=attention.sum(-1,keepdim=True)
        amount=torch.sigmoid(self.strength)*trust
        if self.complex_features:
            gain,phase=update.split(self.feature_dim,-1)
            gain=0.10*amount*gain;phase=0.05*amount*phase
            # Do not rotate the DC or terminal rFFT bin.
            phase_mask=torch.ones_like(phase);phase_mask[:,0]=0;phase_mask[:,-1]=0
            phase=phase*phase_mask
            original=features.reshape(shape[0],-1,self.feature_dim)
            result=original*(1+gain)*torch.complex(torch.cos(phase),torch.sin(phase))
            memory.modulation_cost=memory.modulation_cost+gain.square().mean()+phase.square().mean()
            return result.reshape(features.shape)
        token_scale=tokens.square().mean(-1,keepdim=True).sqrt().detach().clamp_min(.01)
        delta=.10*amount*update*token_scale
        memory.modulation_cost=memory.modulation_cost+(delta/token_scale).square().mean()
        return (tokens+delta).reshape(shape)


class ConditionalCoupling(nn.Module):
    def __init__(self,horizon,width,parity):
        super().__init__()
        self.register_buffer('source',((torch.arange(horizon)+parity)%2==0).float()[None])
        self.context=nn.Linear(width,width);self.graph_key=nn.Linear(width,16,bias=False)
        self.graph_query=nn.Linear(width,16,bias=False);self.local=nn.Conv1d(1,width,3,padding=1)
        self.graph=nn.Linear(1,width);self.out=nn.Sequential(nn.SiLU(),nn.Linear(width,2))
        nn.init.zeros_(self.out[-1].weight);nn.init.zeros_(self.out[-1].bias)

    def parameters_for(self,x,context):
        values=x*self.source
        scores=self.graph_query(context)@self.graph_key(context).transpose(-1,-2)/4
        scores=scores.masked_fill(self.source[:,None]==0,-1e4)
        edges=scores.softmax(-1)*self.source[:,None]
        edges=edges/edges.sum(-1,keepdim=True).clamp_min(1e-8)
        ctx=self.context(context)
        if x.ndim==3:
            b,s,h=x.shape
            global_source=torch.einsum('bij,bsj->bsi',edges,values)
            local=self.local(values.reshape(b*s,1,h)).transpose(1,2).reshape(b,s,h,-1)
            hidden=ctx[:,None]+local+self.graph(global_source[...,None])
        else:
            global_source=edges@values[...,None]
            hidden=ctx+self.local(values[:,None]).transpose(1,2)+self.graph(global_source)
        shift,log_scale=self.out(hidden).unbind(-1)
        destination=1-self.source
        return .35*torch.tanh(shift)*destination,.40*torch.tanh(log_scale)*destination

    def normalize(self,x,context):
        shift,log_scale=self.parameters_for(x,context)
        return (x-shift)*(-log_scale).exp(),-log_scale.sum(-1)

    def inverse(self,z,context):
        shift,log_scale=self.parameters_for(z,context)
        return z*log_scale.exp()+shift


class TrajectoryFlow(nn.Module):
    def __init__(self,horizon,width=64,layers=4,samples=16):
        super().__init__()
        self.layers=nn.ModuleList([ConditionalCoupling(horizon,width,i%2) for i in range(layers)])
        engine=torch.quasirandom.SobolEngine(horizon,scramble=True,seed=2026)
        u=engine.draw(samples//2).clamp(1e-5,1-1e-5)
        z=math.sqrt(2)*torch.erfinv(2*u-1)
        self.register_buffer('noise',torch.cat((z,-z),0))

    def nll(self,residual,context):
        z=residual;jac=z.new_zeros(len(z))
        for layer in self.layers:
            z,value=layer.normalize(z,context);jac=jac+value
        return (.5*(z.square()+math.log(2*math.pi))).sum(-1)-jac

    def samples(self,context):
        z=self.noise[None].expand(context.shape[0],-1,-1)
        # Context graph is shared across samples: no S-fold HxH replication.
        for layer in reversed(self.layers):z=layer.inverse(z,context)
        return z


def risk_decision(draws,weights,mae_weight=.5,smoothing=.02):
    """Empirical Bayes action for squared + smoothed absolute error.

    Bisection computes the unique root. Implicit differentiation gives gradients
    into mixture probabilities and flow samples, rather than an unrelated head.
    Draws are standardized target values; smoothing is in that same scale.
    """
    mean=(draws*weights).sum(1)
    if not mae_weight:return mean
    def equation(a):
        diff=a[:,None]-draws
        norm=(diff.square()+smoothing*smoothing).sqrt()
        f=2*(a-mean)+mae_weight*(weights*diff/norm).sum(1)
        derivative=2+mae_weight*(weights*smoothing*smoothing/norm.pow(3)).sum(1)
        return f,derivative
    with torch.no_grad():
        lo=mean-mae_weight/2;hi=mean+mae_weight/2
        for _ in range(24):
            mid=(lo+hi)/2;f,_=equation(mid)
            lo=torch.where(f<0,mid,lo);hi=torch.where(f<0,hi,mid)
        root=(lo+hi)/2
    if torch.is_grad_enabled():
        f,derivative=equation(root)
        return root-(f-f.detach())/derivative.detach()
    return root


class SemanticDistribution(nn.Module):
    def __init__(self,horizon,feature_dim,width=64):
        super().__init__();self.horizon=horizon
        self.memory=EventMemory(width);self.hidden_projection=nn.Linear(feature_dim,width)
        self.forecast_projection=nn.Linear(3,width)
        self.context=nn.Sequential(nn.Linear(width,width),nn.LayerNorm(width),nn.SiLU())
        self.flow=TrajectoryFlow(horizon,width)
        self.reliability=nn.Sequential(nn.Linear(2*width+1,width),nn.SiLU(),nn.Linear(width,1))
        nn.init.zeros_(self.reliability[-1].weight);nn.init.constant_(self.reliability[-1].bias,-2.0)
        self.state=None

    def finish(self,native_prediction,hidden):
        if torch.is_complex(hidden):hidden=torch.cat((hidden.real,hidden.imag),-1)
        tokens=hidden.reshape(hidden.shape[0],-1,hidden.shape[-1])
        h=self.horizon;scale=self.memory.scale
        trajectory=(native_prediction.detach()-self.memory.level)/scale
        slope=F.pad(trajectory[:,1:]-trajectory[:,:-1],(0,0,1,0))
        t=torch.linspace(0,1,h,device=hidden.device)[None,:,None].expand(hidden.shape[0],-1,-1)
        q=self.memory.numeric_states[:,-1:, :]+self.hidden_projection(tokens.mean(1))[:,None]
        q=q+self.forecast_projection(torch.cat((trajectory,slope,t),-1))
        q=q+positions(h,q.shape[-1],q.device,start=self.memory.numeric_states.shape[1])[None]
        events,attention=self.memory.lookup(q)
        coverage=self.memory.valid.float().mean(1)[:,None,None].expand(-1,h,1)
        gate=torch.sigmoid(self.reliability(torch.cat((q.detach(),events.detach(),coverage),-1))).mean(1)
        gate=(gate*self.memory.valid.any(-1)[:,None]).expand(-1,h)
        native_context=self.context(q);semantic_context=self.context(q+events)
        native_draws=self.flow.samples(native_context);semantic_draws=self.flow.samples(semantic_context)
        count=native_draws.shape[1]
        residual_draws=torch.cat((native_draws,semantic_draws),1)
        weights=torch.cat(((1-gate)[:,None].expand(-1,count,-1)/count,
                           gate[:,None].expand(-1,count,-1)/count),1)
        absolute=native_prediction.squeeze(-1)[:,None]+scale.squeeze(-1)[:,None]*residual_draws
        action=risk_decision(absolute,weights)
        self.state=dict(base=native_prediction,prediction=action[...,None],draws=absolute,
            residual_draws=residual_draws,weights=weights,gate=gate,context=native_context,
            semantic_context=semantic_context,attention=attention,mean=(absolute*weights).sum(1))
        return action[...,None]

    def density(self,target,detach_context=False):
        st=self.state
        residual=((target-st['base'].detach())/self.memory.scale).squeeze(-1)
        cn=st['context'].detach() if detach_context else st['context']
        cs=st['semantic_context'].detach() if detach_context else st['semantic_context']
        native_nll=self.flow.nll(residual,cn);semantic_nll=self.flow.nll(residual,cs)
        # One mixture probability for the whole trajectory, also used in every
        # marginal decision. Thus density, sampling weights and outputs agree.
        gate=st['gate'].mean(-1).clamp(1e-6,1-1e-6)
        if detach_context:gate=gate.detach()
        mixed=-torch.logaddexp(torch.log1p(-gate)-native_nll,gate.log()-semantic_nll)
        mixed=mixed+self.horizon*self.memory.scale[:,0,0].log()
        return mixed,native_nll,semantic_nll

    def objective(self,target,nll_weight=.01,mae_weight=.5,energy_weight=.005,regularity_weight=.02):
        st=self.state;error=st['prediction']-target
        point=error.square().mean()+mae_weight*error.abs().mean()
        nll,nnll,snll=self.density(target,detach_context=True)
        nll=(nll/self.horizon).mean()
        draws=st['draws'];w=st['weights']
        energy=((draws-target.squeeze(-1)[:,None]).abs()*w).sum(1).mean()
        pair=(draws[:,:,None]-draws[:,None,:]).abs()
        energy=energy-.5*(pair*w[:,:,None]*w[:,None,:]).sum((1,2)).mean()
        utility_gain=((nnll-snll)/self.horizon-.02).clamp_min(0)
        teacher=(1-torch.exp(-utility_gain/.2)).detach()
        active=self.memory.valid.any(-1).float()
        utility=F.binary_cross_entropy(st['gate'].mean(-1).clamp(1e-6,1-1e-6),teacher,reduction='none')
        utility=(utility*active).sum()/active.sum().clamp_min(1)
        regularity=(st['prediction']-st['base']).square().mean()
        feature_cost=self.memory.modulation_cost
        loss=point+nll_weight*nll+energy_weight*energy+regularity_weight*regularity+.002*utility+.005*feature_cost
        return loss,dict(point=float(point.detach()),nll=float(nll.detach()),energy=float(energy.detach()),
            utility=float(utility.detach()),regularity=float(regularity.detach()),feature_cost=float(feature_cost.detach()))

    def summaries(self):
        st=self.state;draws=st['draws'];w=st['weights']
        sorted_draws,order=draws.sort(dim=1);cdf=w.gather(1,order).cumsum(1)
        def quantile(p):
            idx=(cdf<p).sum(1).clamp_max(draws.shape[1]-1)
            return sorted_draws.gather(1,idx[:,None]).squeeze(1)
        count=draws.shape[1]//2
        reference=risk_decision(draws[:,:count],torch.ones_like(w[:,:count])/count)
        return dict(distribution_mean=st['mean'],p05=quantile(.05),p50=quantile(.5),p95=quantile(.95),
            text_gate=st['gate'],text_effect=st['prediction'].squeeze(-1)-reference,
            probability_above_last=((draws>self.memory.last_observation.squeeze(-1)[:,None]).float()*w).sum(1))

    def release(self):
        self.state=None
        for name in ('level','scale','numeric_states','event_states','valid','position','gate','last_observation','modulation_cost'):
            if hasattr(self.memory,name):delattr(self.memory,name)
