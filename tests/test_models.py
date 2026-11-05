import torch
from rimeskg.models import KGCModel


def test_T19_complex_is_directional_and_distmult_symmetric():
    z=torch.zeros(3,4)
    rel=torch.zeros(1,4)
    z[0,:]=torch.tensor([1.,0.,0.,1.]);z[1,:]=torch.tensor([0.,1.,1.,0.])
    rel[0,:]=torch.tensor([0.,0.,1.,0.])
    c=KGCModel(3,1,2,encoder="static",decoder="complex")
    with torch.no_grad():c.entities.weight.copy_(z);c.relations.weight.copy_(rel)
    a=c.score(torch.tensor([[0,0,1]]),c.encode(None))
    b=c.score(torch.tensor([[1,0,0]]),c.encode(None))
    assert (a-b).abs().item()>1.e-4
    d=KGCModel(3,1,2,encoder="static",decoder="distmult")
    t1=d.score(torch.tensor([[0,0,1]]),d.encode(None))
    t2=d.score(torch.tensor([[1,0,0]]),d.encode(None))
    assert torch.allclose(t1,t2)


def test_full_candidate_logits_exact():
    torch.manual_seed(8)
    for decoder in ("complex","distmult"):
        m=KGCModel(9,3,4,encoder="static",decoder=decoder)
        z=m.encode(None)
        queries=torch.tensor([[0,0,2],[3,1,5],[4,2,1]])
        for direction in ("head","tail"):
            bulk=m.candidate_scores(queries[:,0],queries[:,1],queries[:,2],z,direction)
            singles=[]
            for h,r,t in queries:
                candidates=torch.tensor([[int(h),int(r),i] if direction=="tail" else [i,int(r),int(t)] for i in range(9)])
                singles.append(m.score(candidates,z))
            assert torch.allclose(bulk,torch.stack(singles),atol=1e-6),decoder+direction


def test_T21_graph_dep_and_static_invariance(toy):
    train,*_=toy
    graph=torch.as_tensor(train)
    g2=graph[:6]
    torch.manual_seed(11)
    m=KGCModel(10,3,8,encoder="rgcn",decoder="complex",layers=2)
    m.eval()
    with torch.no_grad():
        z1=m.encode(graph);z2=m.encode(g2)
        assert not torch.allclose(z1,z2)
        q=torch.tensor([[0,1,2],[6,2,7]])
        assert not torch.allclose(m.score(q,z1),m.score(q,z2))
        assert torch.allclose(m.encode(graph,graph_off=True),m.encode(g2,graph_off=True))
    m2=KGCModel(10,3,8,encoder="static",decoder="complex")
    assert torch.allclose(m2.encode(graph),m2.encode(g2))
